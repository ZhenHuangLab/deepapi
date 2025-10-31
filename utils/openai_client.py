"""
OpenAI 客户端包装器
用于调用后端 LLM 提供商
"""
from typing import Optional, Dict, Any, List, AsyncIterator
from openai import AsyncOpenAI
import json
import logging
import httpx
from models import extract_text_from_content

logger = logging.getLogger(__name__)


class OpenAIClient:
    """OpenAI 客户端包装器"""

    def __init__(self, base_url: str, api_key: str, rpm: Optional[int] = None, max_retry: int = 3, use_responses_api: bool = False):
        # OpenAI客户端自己会管理连接，不需要我们操心
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            max_retries=max_retry,
        )
        self.rpm = rpm
        self.rate_limiter = None
        self.use_responses_api = use_responses_api

        # 统计信息
        self.api_calls = 0
        self.total_tokens = 0
        self.prompt_tokens_total = 0
        self.completion_tokens_total = 0

        # 如果设置了RPM限制，导入限流器
        if rpm:
            from utils.rate_limiter import rate_limiter
            self.rate_limiter = rate_limiter

    def get_statistics(self) -> Dict[str, int]:
        """获取统计信息"""
        return {
            "api_calls": self.api_calls,
            "total_tokens": self.total_tokens,
        }

    def get_usage(self) -> Dict[str, int]:
        """获取更细粒度的 Token 使用情况"""
        return {
            "prompt_tokens": self.prompt_tokens_total,
            "completion_tokens": self.completion_tokens_total,
            "total_tokens": self.total_tokens,
        }

    def _to_text(self, content: Any) -> str:
        try:
            return extract_text_from_content(content)
        except Exception:
            if isinstance(content, str):
                return content
            try:
                return json.dumps(content, ensure_ascii=False)
            except Exception:
                return str(content)

    async def generate_text(
        self,
        model: str,
        prompt: str = None,
        messages: List[Dict[str, Any]] = None,
        system: str = None,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        生成文本
        支持多模态内容（文本和图片）
        """
        # RPM限制 - 在调用后端API之前等待
        if self.rate_limiter and self.rpm:
            await self.rate_limiter.wait_for_rate_limit(
                f"backend_api_{model}",
                self.rpm,
                60
            )

        # 构建消息列表
        if messages is None:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            if prompt is not None:
                messages.append({"role": "user", "content": prompt})
        else:
            if system:
                messages = [{"role": "system", "content": system}] + messages

        try:
            if self.use_responses_api:
                # 使用 Responses API（强制使用流式）
                # Responses API 要求 input 为列表格式
                input_list = messages if messages else []
                kwargs_payload: Dict[str, Any] = dict(kwargs)
                if max_tokens is not None:
                    kwargs_payload["max_output_tokens"] = max_tokens

                # 使用流式响应
                stream = await self.client.responses.create(
                    model=model,
                    input=input_list,
                    temperature=temperature,
                    stream=True,
                    **kwargs_payload
                )

                # 收集流式响应
                chunks: List[str] = []
                async for event in stream:
                    # 尝试从流式响应中提取文本
                    text_chunk = None
                    try:
                        t = getattr(event, "type", "")
                        # 标准 Responses API 事件
                        if t == "response.output_text.delta":
                            delta = getattr(event, "delta", "")
                            if delta:
                                text_chunk = delta
                        # 兼容某些实现直接暴露 delta/text/content 字段
                        elif hasattr(event, 'delta'):
                            delta = getattr(event, 'delta')
                            if isinstance(delta, str):
                                text_chunk = delta
                            else:
                                maybe_text = getattr(delta, 'text', None)
                                if maybe_text:
                                    text_chunk = maybe_text
                                else:
                                    content = getattr(delta, 'content', None)
                                    if isinstance(content, str):
                                        text_chunk = content
                                    elif isinstance(content, list) and len(content) > 0:
                                        part = content[0]
                                        if hasattr(part, 'text') and getattr(part, 'text'):
                                            text_chunk = part.text
                                        elif isinstance(part, dict) and 'text' in part:
                                            text_chunk = part['text']
                        elif hasattr(event, 'output_text'):
                            text_chunk = getattr(event, 'output_text')
                        elif hasattr(event, 'output'):
                            output = getattr(event, 'output')
                            if output and len(output) > 0:
                                first = output[0]
                                content = getattr(first, 'content', None)
                                if content and len(content) > 0:
                                    part = content[0]
                                    if hasattr(part, 'text') and getattr(part, 'text'):
                                        text_chunk = part.text
                                    elif isinstance(part, dict) and 'text' in part:
                                        text_chunk = part['text']
                    except Exception as e:
                        logger.debug(f"Error extracting text from responses stream event: {e}")

                    if text_chunk:
                        chunks.append(text_chunk)

                # 统计 API 调用
                self.api_calls += 1

                return "".join(chunks)
            else:
                # 使用 Chat Completions API（优先走流式，兼容强制 stream 的上游）
                try:
                    stream = await self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=True,
                        **kwargs
                    )
                    chunks: List[str] = []
                    async for chunk in stream:
                        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                            chunks.append(chunk.choices[0].delta.content)
                    # 统计 API 调用（流式无法可靠拿到 usage，这里仅计数）
                    self.api_calls += 1
                    return "".join(chunks)
                except Exception:
                    # 流式不可用时回退到非流式
                    response = await self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **kwargs
                    )

                    # 统计 API 调用
                    self.api_calls += 1
                    if hasattr(response, 'usage') and response.usage:
                        pt = getattr(response.usage, 'prompt_tokens', 0) or 0
                        ct = getattr(response.usage, 'completion_tokens', 0) or 0
                        tt = getattr(response.usage, 'total_tokens', pt + ct) or (pt + ct)
                        self.prompt_tokens_total += pt
                        self.completion_tokens_total += ct
                        self.total_tokens += tt

                    if not response.choices or len(response.choices) == 0:
                        logger.error(f"API 返回空响应: model={model}, response={response}")
                        raise ValueError(f"API 返回空响应，可能是模型过载或请求被拒绝。模型: {model}")

                    return response.choices[0].message.content

        except Exception as e:
            # 某些上游要求强制流式
            if "Stream must be set to true" in str(e):
                if self.use_responses_api:
                    # 用 Responses API 走流式并聚合
                    chunks: List[str] = []
                    try:
                        async with self.client.responses.stream(
                            model=model,
                            input=messages if messages else [],
                            temperature=temperature,
                            **({ "max_output_tokens": max_tokens } if max_tokens is not None else {}),
                            **kwargs
                        ) as stream:
                            async for event in stream:
                                if getattr(event, "type", "") == "response.output_text.delta":
                                    delta = getattr(event, "delta", "")
                                    if delta:
                                        chunks.append(delta)
                        # 统计 usage（若可用）
                        try:
                            fr = getattr(stream, "final_response", None)
                            usage = getattr(fr, "usage", None) if fr else None
                            if usage:
                                pt = getattr(usage, "input_tokens", 0) or 0
                                ct = getattr(usage, "output_tokens", 0) or 0
                                tt = getattr(usage, "total_tokens", pt + ct) or (pt + ct)
                                self.prompt_tokens_total += pt
                                self.completion_tokens_total += ct
                                self.total_tokens += tt
                        except Exception:
                            pass
                    except Exception:
                        pass
                    return "".join(chunks)
                else:
                    # Chat Completions 走流式补救
                    stream = await self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=True,
                        **kwargs
                    )
                    chunks: List[str] = []
                    async for chunk in stream:
                        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                            chunks.append(chunk.choices[0].delta.content)
                    return "".join(chunks)
            raise

    async def generate_object(
        self,
        model: str,
        prompt: str,
        response_format: Dict[str, Any],
        temperature: float = 1.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        生成结构化对象(JSON)

        Args:
            model: 模型名称
            prompt: 提示词
            response_format: 响应格式定义
            temperature: 温度参数

        Returns:
            解析后的JSON对象
        """
        # RPM限制 - 在调用后端API之前等待
        if self.rate_limiter and self.rpm:
            await self.rate_limiter.wait_for_rate_limit(
                f"backend_api_{model}",
                self.rpm,
                60
            )

        if self.use_responses_api:
            # Responses API（强制使用流式）
            kwargs_payload: Dict[str, Any] = dict(kwargs)
            if response_format:
                kwargs_payload["response_format"] = {"type": "json_object"}

            # 使用流式响应，input 必须是列表格式
            input_list = [{"role": "user", "content": prompt}]
            stream = await self.client.responses.create(
                model=model,
                input=input_list,
                temperature=temperature,
                stream=True,
                **kwargs_payload
            )

            # 收集流式响应
            chunks: List[str] = []
            async for chunk in stream:
                # 尝试从流式响应中提取文本
                text_chunk = None
                try:
                    # 尝试获取 delta 或 output_text
                    if hasattr(chunk, 'delta'):
                        delta = chunk.delta
                        if hasattr(delta, 'text'):
                            text_chunk = delta.text
                        elif hasattr(delta, 'content'):
                            content = delta.content
                            if isinstance(content, str):
                                text_chunk = content
                            elif isinstance(content, list) and len(content) > 0:
                                part = content[0]
                                if hasattr(part, 'text'):
                                    text_chunk = part.text
                                elif isinstance(part, dict) and 'text' in part:
                                    text_chunk = part['text']
                    elif hasattr(chunk, 'output_text'):
                        text_chunk = chunk.output_text
                    elif hasattr(chunk, 'output'):
                        output = chunk.output
                        if output and len(output) > 0:
                            first = output[0]
                            if hasattr(first, 'content'):
                                content = first.content
                                if content and len(content) > 0:
                                    part = content[0]
                                    if hasattr(part, 'text'):
                                        text_chunk = part.text
                                    elif isinstance(part, dict) and 'text' in part:
                                        text_chunk = part['text']
                except Exception as e:
                    logger.debug(f"Error extracting text from chunk: {e}")

                if text_chunk:
                    chunks.append(text_chunk)

            # 统计 API 调用
            self.api_calls += 1

            text = "".join(chunks)
        else:
            # Chat Completions API
            response = await self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                response_format={"type": "json_object"},
                **kwargs
            )

            # 统计 API 调用
            self.api_calls += 1
            if hasattr(response, 'usage') and response.usage:
                pt = getattr(response.usage, 'prompt_tokens', 0) or 0
                ct = getattr(response.usage, 'completion_tokens', 0) or 0
                tt = getattr(response.usage, 'total_tokens', pt + ct) or (pt + ct)
                self.prompt_tokens_total += pt
                self.completion_tokens_total += ct
                self.total_tokens += tt

            # 检查响应是否包含 choices
            if not response.choices or len(response.choices) == 0:
                logger.error(f"API 返回空响应: model={model}, response={response}")
                raise ValueError(f"API 返回空响应，可能是模型过载或请求被拒绝。模型: {model}")

            text = response.choices[0].message.content

        try:
            return json.loads(text or "")
        except json.JSONDecodeError:
            # 尝试从代码块中提取JSON
            if text and "```json" in text:
                json_text = text.split("```json")[1].split("```")[0].strip()
            elif text and "```" in text:
                json_text = text.split("```")[1].split("```")[0].strip()
            else:
                json_text = (text or "").strip()

            return json.loads(json_text)

    async def stream_text(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        流式生成文本
        支持多模态内容（文本和图片）

        Args:
            model: 模型名称
            messages: 消息列表（支持多模态content）
            temperature: 温度参数
            max_tokens: 最大token数

        Yields:
            文本块
        """
        # RPM限制 - 在调用后端API之前等待
        if self.rate_limiter and self.rpm:
            await self.rate_limiter.wait_for_rate_limit(
                f"backend_api_{model}",
                self.rpm,
                60
            )

        stream = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs
        )

        # 统计 API 调用
        self.api_calls += 1

        async for chunk in stream:
            # 检查 chunk 是否包含 choices
            if not chunk.choices or len(chunk.choices) == 0:
                logger.warning(f"流式响应中收到空 chunk: model={model}")
                continue

            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


def create_client(base_url: str, api_key: str, rpm: Optional[int] = None, max_retry: int = 3, use_responses_api: Optional[bool] = None) -> OpenAIClient:
    """创建OpenAI客户端

    Args:
        base_url: 后端基础 URL
        api_key: 后端 API Key
        rpm: 每分钟限流
        max_retry: 最大重试次数
        use_responses_api: 是否使用 Responses API（优先尝试）。
            - True: 优先使用 Responses API（必要时支持流式 fallback）
            - False: 使用 Chat Completions API
            - None: 默认与 False 等价（向后兼容）
    """
    return OpenAIClient(base_url, api_key, rpm, max_retry, use_responses_api if use_responses_api is not None else False)

