import aiohttp
import asyncio
import logging
import random
from typing import Optional, Dict, Any, Iterable


class CallbackClient:
    def __init__(self, url: str, secret: Optional[str]):
        self.url = url
        self.secret = secret
        self._session: Optional[aiohttp.ClientSession] = None
        self._lock = asyncio.Lock()

    async def _get_session(self) -> aiohttp.ClientSession:
        async with self._lock:
            if self._session is None or self._session.closed:
                self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=23))
            return self._session

    def _iter_backoff(self) -> Iterable[float]:
        for base in (0.5, 1.0, 2.0, 4.0, 8.0):
            jitter = random.uniform(0.8, 1.2)
            yield base * jitter

    async def post(self, payload: Dict[str, Any]) -> None:
        headers = {"Content-Type": "application/json"}
        if self.secret:
            headers["x-ai-key"] = self.secret

        for attempt, delay in enumerate(self._iter_backoff(), start=1):
            try:
                s = await self._get_session()
                async with s.post(self.url, json=payload, headers=headers) as resp:
                    await resp.release()
                    if resp.status < 400:
                        logging.getLogger("stt").info(
                            "webhook ok session_id=%s seq=%s status=%s",
                            payload.get("session_id"),
                            payload.get("seq"),
                            resp.status,
                        )
                        return
                    if resp.status not in (408, 429) and resp.status < 500:
                        logging.getLogger("stt").warning(
                            "webhook non-retryable status=%s session_id=%s seq=%s",
                            resp.status,
                            payload.get("session_id"),
                            payload.get("seq"),
                        )
                        return
            except (asyncio.TimeoutError, aiohttp.ClientError):
                pass

            if attempt == 5:
                logging.getLogger("stt").error(
                    "webhook failed after retries session_id=%s seq=%s",
                    payload.get("session_id"),
                    payload.get("seq"),
                )
                return
            await asyncio.sleep(delay)

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
