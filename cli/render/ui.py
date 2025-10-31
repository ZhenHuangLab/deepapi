from typing import Any, Dict
import time

class BaseUI:
    def show_header(self, model: str, temperature: float, max_tokens: int) -> None:
        print(f"[模型] {model}  [温度] {temperature}  [最大tokens] {max_tokens}")

    def add_reasoning_delta(self, text: str) -> None:
        print(f"🔎 {text}", end="", flush=True)

    def add_progress(self, label: str, data: Dict[str, Any]) -> None:
        ts = time.strftime("%H:%M:%S")
        print(f"\n⏱️ {ts} {label}: {data}")

    def add_answer_delta(self, text: str) -> None:
        print(text, end="", flush=True)

    def set_usage(self, usage: Dict[str, int]) -> None:
        print(f"\n📊 Tokens: in={usage.get('input_tokens',0)} out={usage.get('output_tokens',0)} total={usage.get('total_tokens',0)}")

    def finish(self) -> None:
        print("\n--- 完成 ---\n")


def get_ui() -> BaseUI:
    # 预留后续支持 rich 的实现
    return BaseUI()

