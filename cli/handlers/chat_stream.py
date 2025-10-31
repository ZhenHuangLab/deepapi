from typing import Any, Dict, Optional

from cli.render.ui import BaseUI

class ChatStreamHandler:
    def __init__(self, ui: BaseUI) -> None:
        self.ui = ui
        self.answer = []

    def handle(self, evt: Dict[str, Any]) -> Optional[str]:
        if evt.get("object") == "chat.completion.chunk":
            delta = (evt.get("choices") or [{}])[0].get("delta", {})
            if "reasoning_content" in delta:
                self.ui.add_reasoning_delta(delta["reasoning_content"])
            if "content" in delta and delta["content"]:
                self.answer.append(delta["content"])
                self.ui.add_answer_delta(delta["content"])
            if (evt.get("choices") or [{}])[0].get("finish_reason") == "stop":
                return "".join(self.answer)
        return None

