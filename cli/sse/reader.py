import json
from typing import AsyncIterator, Optional, Tuple, Union, Any

DONE = object()

async def aiter_sse_events(resp) -> AsyncIterator[Union[dict, object]]:
    async for line in resp.aiter_lines():
        if not line:
            continue
        if line.startswith("data: "):
            data = line[len("data: "):].strip()
            if data == "[DONE]":
                yield DONE
                break
            try:
                yield json.loads(data)
            except Exception:
                # 忽略坏行
                continue

