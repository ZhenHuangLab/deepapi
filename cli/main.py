import asyncio
import time
from typing import Any, Dict
import httpx


from cli.core.config import load_config
from cli.core.state import ConversationState
from cli.api.client import DeepAPIClient
from cli.handlers.responses_stream import ResponsesStreamHandler
from cli.render.ui import get_ui
from cli.commands import handle_command

async def ainput(prompt: str) -> str:
    return await asyncio.get_event_loop().run_in_executor(None, lambda: input(prompt))

async def run_repl() -> None:
    cfg = load_config()
    client = DeepAPIClient(cfg.base_url, cfg.api_key)
    state = ConversationState(model_id=cfg.deepthink_model_id, mode="deepthink")
    ui = get_ui()

    print("DeepAPI CLI 已启动。输入 /help 查看命令。\n")

    while True:
        line = (await ainput("> ")).rstrip("\n")
        if not line:
            continue
        if line.startswith("/"):
            try:
                cont = await handle_command(line, state, client, cfg)
            except SystemExit:
                print("Bye.")
                return
            except Exception as e:
                print("命令执行出错:", e)
            continue

        if not state.model_id:
            print("请先通过 /model <id> 或 /mode 选择模型。")
            continue

        # 发送消息并流式接收
        state.append_user(line)
        body: Dict[str, Any] = {
            "model": state.model_id,
            "messages": state.build_messages(),
            "temperature": state.temperature,
            "max_tokens": state.max_tokens,
        }
        ui.show_header(state.model_id, state.temperature, state.max_tokens)
        handler = ResponsesStreamHandler(ui)
        try:
            got_any = False
            async for evt in client.stream_responses(body):
                got_any = True
                final = handler.handle(evt)
                if final is not None:
                    answer = final
                    break
            else:
                if not got_any:
                    print("\n[警告] 未收到任何流事件，可能是认证失败、模型ID不存在或端点未以SSE返回。\n")
                answer = ""  # 未正常结束
        except KeyboardInterrupt:
            print("\n(已取消)\n")
            answer = ""
        except httpx.HTTPStatusError as e:
            print(f"\n[请求失败] {e}\n")
            answer = ""
        except Exception as e:
            print(f"\n[异常] {e}\n")
            answer = ""
        ui.finish()
        if answer:
            state.append_assistant(answer)

if __name__ == "__main__":
    asyncio.run(run_repl())

