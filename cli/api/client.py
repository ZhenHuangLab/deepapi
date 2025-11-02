import asyncio
import json
from typing import AsyncIterator, Dict, Any, List, Optional

import httpx

from cli.sse.reader import aiter_sse_events, DONE

class DeepAPIClient:
    def __init__(self, base_url: str, api_key: Optional[str] = None) -> None:
        self.base = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    async def list_models(self) -> List[str]:
        url = f"{self.base}/v1/models"
        async with httpx.AsyncClient(timeout=30, trust_env=False) as s:
            r = await s.get(url, headers=self.headers)
            r.raise_for_status()
            data = r.json()
            # 兼容 OpenAI 风格和简单数组
            if isinstance(data, dict) and "data" in data:
                return [m.get("id") for m in data.get("data", []) if m.get("id")]
            if isinstance(data, list):
                return [str(x) for x in data]
            return []

    async def stream_responses(self, body: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        url = f"{self.base}/v1/responses"
        body = {**body, "stream": True}
        headers = {**self.headers, "Accept": "text/event-stream"}
        async with httpx.AsyncClient(timeout=None, trust_env=False) as s:
            async with s.stream("POST", url, headers=headers, json=body) as r:
                # 显式检查非 2xx，避免静默无输出
                if r.status_code >= 400:
                    raw = await r.aread()
                    text = raw.decode("utf-8", errors="ignore")
                    try:
                        j = json.loads(text)
                        detail = j.get("detail") or j
                    except Exception:
                        detail = text
                    raise httpx.HTTPStatusError(
                        f"{r.status_code} {r.reason_phrase}: {detail}", request=r.request, response=r
                    )
                async for evt in aiter_sse_events(r):
                    if evt is DONE:
                        break
                    yield evt  # dict 事件

    async def stream_chat(self, body: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        url = f"{self.base}/v1/chat/completions"
        body = {**body, "stream": True}
        headers = {**self.headers, "Accept": "text/event-stream"}
        async with httpx.AsyncClient(timeout=None, trust_env=False) as s:
            async with s.stream("POST", url, headers=headers, json=body) as r:
                if r.status_code >= 400:
                    raw = await r.aread()
                    text = raw.decode("utf-8", errors="ignore")
                    try:
                        j = json.loads(text)
                        detail = j.get("detail") or j
                    except Exception:
                        detail = text
                    raise httpx.HTTPStatusError(
                        f"{r.status_code} {r.reason_phrase}: {detail}", request=r.request, response=r
                    )
                async for evt in aiter_sse_events(r):
                    if evt is DONE:
                        break
                    yield evt

