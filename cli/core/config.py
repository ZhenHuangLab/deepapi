import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict

CONFIG_PATH = Path(os.path.expanduser("~/.deepapi-cli/config.json"))

@dataclass
class CLIConfig:
    base_url: str = os.getenv("DEEPAPI_BASE_URL", "http://localhost:8000")
    api_key: Optional[str] = os.getenv("DEEPAPI_API_KEY")
    deepthink_model_id: str = os.getenv("DEEPAPI_DEEPTHINK_MODEL", "gemini-2.5-pro-deepthink")
    ultrathink_model_id: str = os.getenv("DEEPAPI_ULTRATHINK_MODEL", "gpt-4o-ultrathink")

    def headers(self) -> Dict[str, str]:
        h: Dict[str, str] = {}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

def load_config() -> CLIConfig:
    cfg = CLIConfig()
    if CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text())
            for k, v in data.items():
                if hasattr(cfg, k) and v is not None:
                    setattr(cfg, k, v)
        except Exception:
            pass
    return cfg

def save_config(cfg: CLIConfig) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2))

