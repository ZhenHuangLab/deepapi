from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class ConversationState:
    messages: List[Dict[str, str]] = field(default_factory=list)
    system_prompt: Optional[str] = None
    model_id: Optional[str] = None
    mode: str = "deepthink"
    temperature: float = 0.7
    max_tokens: int = 2048

    def build_messages(self) -> List[Dict[str, str]]:
        msgs: List[Dict[str, str]] = []
        if self.system_prompt:
            msgs.append({"role": "system", "content": self.system_prompt})
        msgs.extend(self.messages)
        return msgs

    def append_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def append_assistant(self, text: str) -> None:
        self.messages.append({"role": "assistant", "content": text})

