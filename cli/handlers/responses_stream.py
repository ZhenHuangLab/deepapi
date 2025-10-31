import time
from typing import Any, Dict, Optional

from cli.render.ui import BaseUI

EVENT_LABEL = {
    "init": "开始分析",
    "planning": "生成计划",
    "thinking": "思考中",
    "verification": "验证",
    "correction": "修正",
    "summarizing": "总结",
    "success": "成功",
    "failure": "失败",
    "agent-update": "Agent 更新",
}

class ResponsesStreamHandler:
    def __init__(self, ui: BaseUI) -> None:
        self.ui = ui
        self.answer = []
        self.reasoning = []
        self.start_ts = time.time()

    def handle(self, evt: Dict[str, Any]) -> Optional[str]:
        t = evt.get("type")
        if t == "response.created":
            self.start_ts = time.time()
        elif t == "response.progress":
            p = evt.get("progress", {})
            name = p.get("event") or "progress"
            label = EVENT_LABEL.get(name, name)
            self.ui.add_progress(label, p)
        elif t == "response.reasoning.delta":
            d = evt.get("delta", "")
            if d:
                self.reasoning.append(d)
                self.ui.add_reasoning_delta(d)
        elif t == "response.output_text.delta":
            d = evt.get("delta", "")
            if d:
                self.answer.append(d)
                self.ui.add_answer_delta(d)
        elif t == "response.output_text.done":
            pass
        elif t == "response.completed":
            resp = evt.get("response", {})
            usage = resp.get("usage")
            if usage:
                self.ui.set_usage(usage)
            return "".join(self.answer)
        return None

