"""
OpenAI Responses 标准接口（/v1/responses）
在不破坏现有 /v1/chat/completions 的前提下，并存支持。
"""
import time
import uuid
import json
import logging
import asyncio
from typing import Any, Dict, List, Optional, AsyncIterator

from fastapi import APIRouter, HTTPException, Header
from fastapi.responses import StreamingResponse

from models import (
    ResponseRequest, Response as ResponsesObject, ResponseUsage,
    Message, extract_text_from_content
)
from config import config
from utils.openai_client import create_client
from engine.deep_think import DeepThinkEngine
from engine.ultra_think import UltraThinkEngine
from utils.summary_think import ThinkingSummaryGenerator, UltraThinkSummaryGenerator
from .chat import verify_auth, extract_llm_params, process_user_messages

router = APIRouter()
logger = logging.getLogger(__name__)


def _coalesce_messages(req: ResponseRequest) -> List[Message]:
    """将 Responses 请求体归一化为 Chat 消息列表。
    支持以下几种输入：
    - req.messages（与 chat-completions 一致）
    - req.input 为字符串：作为一条 user 消息
    - req.input 为列表：
        * 若元素含 role，则按 Chat 消息解释
        * 若元素为 {"type":"input_text","text":...}，则合并为一条 user 文本
    """
    if req.messages:
        return req.messages

    if isinstance(req.input, str):
        return [Message(role="user", content=req.input)]

    if isinstance(req.input, list):
        # 形如 Chat 风格
        if req.input and isinstance(req.input[0], dict) and "role" in req.input[0]:
            msgs: List[Message] = []
            for item in req.input:  # type: ignore[assignment]
                role = item.get("role")
                content = item.get("content")
                if role and content is not None:
                    msgs.append(Message(role=role, content=content))
            if msgs:
                return msgs
        # 形如 input_text 片段
        texts: List[str] = []
        for item in req.input:
            if isinstance(item, dict) and item.get("type") == "input_text":
                txt = item.get("text", "")
                if txt:
                    texts.append(txt)
        if texts:
            return [Message(role="user", content="\n".join(texts))]

    # 兜底：空列表
    return []


async def _stream_responses(engine, request_id: str, created: int, model: str, thinking_generator=None) -> AsyncIterator[str]:
    """
    以 Responses SSE 事件格式流式输出，增加推理/进度事件：
    - response.created
    - response.progress（附带 event/data）
    - response.reasoning.delta（根据进度事件生成的思维文本）
    - response.output_text.delta
    - response.output_text.done（可选）
    - response.completed（包含 usage）
    """
    # 发送 created 事件
    yield f"data: {json.dumps({'type': 'response.created', 'response': {'id': request_id}}, ensure_ascii=False)}\n\n"

    progress_queue = []

    def on_progress(event):
        progress_queue.append(event)

    # 挂载进度回调
    engine.on_progress = on_progress

    # UltraThink 的 agent 更新也映射为 progress 事件
    if hasattr(engine, 'on_agent_update'):
        def on_agent_update(agent_id: str, update: Dict[str, Any]):
            from models import ProgressEvent
            progress_queue.append(ProgressEvent(type='agent-update', data={'agentId': agent_id, **update}))
        engine.on_agent_update = on_agent_update

    # 启动引擎
    engine_task = asyncio.create_task(engine.run())

    import logging
    logger = logging.getLogger(__name__)

    try:
        # 流式发送进度/推理
        while not engine_task.done():
            while progress_queue:
                event = progress_queue.pop(0)

                # DEBUG: 记录事件类型和数据
                event_type = getattr(event, 'type', '')
                event_data = getattr(event, 'data', {}) or {}
                if event_type in ['planning', 'solution']:
                    logger.info(f"[DEBUG] Event type: {event_type}, data keys: {list(event_data.keys())}")
                    if 'plan' in event_data:
                        logger.info(f"[DEBUG] Plan length: {len(event_data.get('plan', ''))}")
                    if 'solution' in event_data:
                        logger.info(f"[DEBUG] Solution length: {len(event_data.get('solution', ''))}")

                # 统一的 progress 事件（结构化）
                evt_progress = {
                    'type': 'response.progress',
                    'progress': {
                        'event': event_type,
                        **event_data
                    },
                    'response': {'id': request_id}
                }
                yield f"data: {json.dumps(evt_progress, ensure_ascii=False)}\n\n"

                # 将进度事件转换为 reasoning 文本增量
                if thinking_generator:
                    thinking_text = thinking_generator.process_event(event)
                    if thinking_text:
                        evt_reasoning = {
                            'type': 'response.reasoning.delta',
                            'delta': thinking_text,
                            'response': {'id': request_id}
                        }
                        yield f"data: {json.dumps(evt_reasoning, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.1)

        # 获取最终结果
        result = await engine_task
        final_text = result.summary or result.final_solution

        # 切块输出最终文本
        for i in range(0, len(final_text), 50):
            chunk = final_text[i:i+50]
            evt = {
                'type': 'response.output_text.delta',
                'delta': chunk,
                'response': {'id': request_id},
            }
            yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"

        # 可选:done 标记
        yield f"data: {json.dumps({'type': 'response.output_text.done', 'response': {'id': request_id}}, ensure_ascii=False)}\n\n"

        # usage 统计
        try:
            usage_breakdown = engine.client.get_usage()  # type: ignore[attr-defined]
        except Exception:
            usage_breakdown = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}

        # 完成事件，包含最终 response 对象
        message_id = f"msg-{uuid.uuid4().hex[:8]}"
        response_obj = {
            'id': request_id,
            'object': 'response',
            'created': created,
            'model': model,
            'output': [
                {
                    'id': message_id,
                    'type': 'message',
                    'role': 'assistant',
                    'content': [
                        {'type': 'output_text', 'text': final_text}
                    ],
                }
            ],
            'usage': {
                'input_tokens': usage_breakdown.get('prompt_tokens', 0),
                'output_tokens': usage_breakdown.get('completion_tokens', 0),
                'total_tokens': usage_breakdown.get('total_tokens', 0)
            },
            'status': 'completed',
        }
        evt_completed = {'type': 'response.completed', 'response': response_obj}
        yield f"data: {json.dumps(evt_completed, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.error(f"Responses streaming error: {e}")
        raise


@router.post("/v1/responses")
async def responses_endpoint(
    request: ResponseRequest,
    authorization: str = Header(None)
):
    """
    OpenAI Responses 标准接口，映射至现有 DeepThink/UltraThink 引擎。
    """
    # 验证 API Key
    verify_auth(authorization)

    # 模型配置
    model_config = config.get_model(request.model)
    if not model_config:
        raise HTTPException(status_code=404, detail=f"Model {request.model} not found")
    provider_config = config.get_provider(model_config.provider)
    if not provider_config:
        raise HTTPException(status_code=500, detail=f"Provider {model_config.provider} not configured")

    # 统一提取 LLM 参数（兼容 max_output_tokens -> max_tokens）
    llm_params = extract_llm_params(
        # 临时构造 ChatCompletionRequest 同名字段子集
        type("_Tmp", (), {
            "temperature": request.temperature,
            "max_tokens": request.max_output_tokens if request.max_output_tokens is not None else request.max_tokens,
        })()
    )

    # 归一化消息并应用与 chat-completions 相同的 system 合并策略
    raw_messages = _coalesce_messages(request)
    if not raw_messages:
        raise HTTPException(status_code=400, detail="No input/messages found")
    processed_messages = process_user_messages(raw_messages)

    # 提取当前问题与上下文
    user_messages = [m for m in processed_messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found")
    last_user_message = user_messages[-1]
    problem_statement_raw = last_user_message.content
    problem_statement_text = extract_text_from_content(problem_statement_raw)

    conversation_history: List[Dict[str, Any]] = []
    if len(processed_messages) > 1:
        for m in processed_messages[:-1]:
            conversation_history.append({"role": m.role, "content": m.content})

    # 创建后端客户端
    max_retry = model_config.get_max_retry(default=config.max_retry)
    client = create_client(provider_config.base_url, provider_config.key, model_config.rpm, max_retry, provider_config.response_api)
    # 如果启用了 summary_think,创建思维链生成器（与 chat 接口保持一致）
    thinking_generator = None
    if getattr(model_config, 'has_summary_think', False):
        if model_config.level == 'ultrathink':
            thinking_generator = UltraThinkSummaryGenerator()
        else:
            thinking_generator = ThinkingSummaryGenerator(mode='deepthink')


    # 选择引擎
    if model_config.level == "ultrathink":
        engine = UltraThinkEngine(
            client=client,
            model=model_config.model,
            problem_statement=problem_statement_raw,
            conversation_history=conversation_history,
            max_iterations=model_config.max_iterations,
            required_successful_verifications=model_config.required_verifications,
            num_agents=model_config.num_agent,
            parallel_run_agent=model_config.parallel_run_agent,
            model_stages=model_config.models,
            enable_parallel_check=model_config.parallel_check,
            llm_params=llm_params,
        )
    else:
        engine = DeepThinkEngine(
            client=client,
            model=model_config.model,
            problem_statement=problem_statement_raw,
            conversation_history=conversation_history,
            max_iterations=model_config.max_iterations,
            required_successful_verifications=model_config.required_verifications,
            model_stages=model_config.models,
            enable_planning=model_config.has_plan_mode,
            enable_parallel_check=model_config.parallel_check,
            llm_params=llm_params,
        )

    # 流式与非流式
    request_id = f"resp-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    if request.stream:
        return StreamingResponse(
            _stream_responses(engine, request_id, created, request.model, thinking_generator),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # 非流式：直接返回标准 Responses 对象
    result = await engine.run()
    final_text = result.summary or result.final_solution
    try:
        usage_breakdown = client.get_usage()  # type: ignore[attr-defined]
    except Exception:
        usage_breakdown = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}


    message_id = f"msg-{uuid.uuid4().hex[:8]}"
    response_obj = ResponsesObject(
        id=request_id,
        created=created,
        model=request.model,
        output=[
            {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": final_text}],
            }
        ],
        output_text=final_text,
        usage=ResponseUsage(
            input_tokens=usage_breakdown.get('prompt_tokens', 0),
            output_tokens=usage_breakdown.get('completion_tokens', 0),
            total_tokens=usage_breakdown.get('total_tokens', 0),
        ),
        status="completed",
    )
    return response_obj

