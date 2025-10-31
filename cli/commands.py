from typing import Optional

from cli.core.config import CLIConfig, save_config
from cli.core.state import ConversationState
from cli.api.client import DeepAPIClient

HELP = """
/ help              显示帮助
/ models            列出可用模型 ID
/ model <id>        设置当前模型
/ mode <deepthink|ultrathink> 切换模式（使用本地预设模型ID）
/ new               开启新对话
/ sys <text>        设置/更新系统提示词
/ temp <float>      设置温度
/ maxtokens <int>   设置最大生成 tokens
/ exit              退出
""".strip()

async def handle_command(line: str, state: ConversationState, client: DeepAPIClient, cfg: CLIConfig) -> bool:
    parts = line.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if cmd in ("/help", "/h"):
        print(HELP)
        return True

    if cmd == "/models":
        try:
            ids = await client.list_models()
            print("可用模型:")
            for mid in ids:
                print(" -", mid)
        except Exception as e:
            print("获取模型失败:", e)
        return True

    if cmd == "/model":
        if not arg:
            print("用法: /model <id>")
        else:
            state.model_id = arg.strip()
            print("当前模型:", state.model_id)
        return True

    if cmd == "/mode":
        v = arg.strip().lower()
        if v not in ("deepthink", "ultrathink"):
            print("用法: /mode deepthink|ultrathink")
        else:
            state.mode = v
            state.model_id = cfg.deepthink_model_id if v == "deepthink" else cfg.ultrathink_model_id
            print(f"已切换到 {v}，模型: {state.model_id}")
        return True

    if cmd == "/new":
        state.messages.clear()
        print("已清空会话历史。")
        return True

    if cmd == "/sys":
        state.system_prompt = arg
        print("已设置系统提示词。")
        return True

    if cmd == "/temp":
        try:
            state.temperature = float(arg)
            print("温度=", state.temperature)
        except Exception:
            print("用法: /temp <float>")
        return True

    if cmd == "/maxtokens":
        try:
            state.max_tokens = int(arg)
            print("max_tokens=", state.max_tokens)
        except Exception:
            print("用法: /maxtokens <int>")
        return True

    if cmd == "/exit":
        raise SystemExit(0)

    print("未知命令，输入 /help 查看帮助。")
    return True

