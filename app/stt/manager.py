import asyncio
import logging
import os
import uuid
from typing import Dict, Optional, Any, Literal, List
from pydantic import BaseModel, Field

from app.stt.worker import STTWorker, CaptionEvent
from app.stt.callbacks import CallbackClient


class InputJRTPOpus(BaseModel):
    type: Literal["janus_rtp_opus"] = "janus_rtp_opus"
    listen_ip: str = "0.0.0.0"
    audio_port: int
    payload_type: int = 111
    ssrc: Optional[int] = None


class InputFFmpegURL(BaseModel):
    type: Literal["ffmpeg_url"] = "ffmpeg_url"
    url: str


InputConfig = InputJRTPOpus | InputFFmpegURL


class ASRConfig(BaseModel):
    model_size: Optional[str] = "small"
    language: Optional[str] = "vi"
    vad_filter: bool = True
    beam_size: int = 1
    chunk_seconds: float = 0.6
    overlap_seconds: float = 0.3
    emit_interval_ms: int = 600


class CallbackConfig(BaseModel):
    url: Optional[str] = None
    secret: Optional[str] = None


class StartSessionRequest(BaseModel):
    live_id: str = Field(..., min_length=1)
    room_id: Optional[str] = None
    publisher_id: Optional[str] = None
    input: InputConfig
    asr: ASRConfig = ASRConfig()
    callback: CallbackConfig = CallbackConfig()


class SessionManager:
    def __init__(self, device: str, compute_type: str, default_model_size: str, max_sessions: int):
        self.device = device
        self.compute_type = compute_type
        self.default_model_size = default_model_size
        self.max_sessions = max_sessions

        self._lock = asyncio.Lock()
        self._sessions: Dict[str, STTWorker] = {}
        self._subscribers: Dict[str, List[asyncio.Queue]] = {}

        self._sem = asyncio.Semaphore(max_sessions)

    async def start(self, req: StartSessionRequest) -> str:
        async with self._lock:
            if len(self._sessions) >= self.max_sessions:
                raise ValueError(f"too many active sessions (max {self.max_sessions})")

            if req.input.type == "janus_rtp_opus":
                if "payload_type" not in req.input.model_fields_set:
                    logging.getLogger("stt").warning(
                        "payload_type missing; defaulting to 111 for janus_rtp_opus"
                    )
                if "audio_port" not in req.input.model_fields_set:
                    logging.getLogger("stt").error(
                        "audio_port missing for janus_rtp_opus"
                    )
                if req.input.audio_port <= 0:
                    raise ValueError("audio_port must be a positive integer")
                if req.input.payload_type <= 0:
                    raise ValueError("payload_type must be a positive integer")

            session_id = uuid.uuid4().hex
            model_size = req.asr.model_size or self.default_model_size

            backend_url = os.getenv("BACKEND_URL")
            if backend_url:
                backend_url = backend_url.rstrip("/")
            default_callback_url = f"{backend_url}/janus/api/ai/captions" if backend_url else None
            callback_url = req.callback.url or default_callback_url
            callback_secret = os.getenv("AI_KEY")
            callback_client = CallbackClient(callback_url, callback_secret) if callback_url else None

            logging.getLogger("stt").info(
                "session %s create live_id=%s room_id=%s publisher_id=%s callback_url=%s",
                session_id,
                req.live_id,
                req.room_id,
                req.publisher_id,
                callback_url,
            )

            worker = STTWorker(
                session_id=session_id,
                live_id=req.live_id,
                room_id=req.room_id,
                publisher_id=req.publisher_id,
                input_cfg=req.input,
                model_size=model_size,
                language="vi",
                vad_filter=req.asr.vad_filter,
                beam_size=req.asr.beam_size,
                chunk_seconds=req.asr.chunk_seconds,
                overlap_seconds=req.asr.overlap_seconds,
                emit_interval_ms=req.asr.emit_interval_ms,
                device=self.device,
                compute_type=self.compute_type,
                sem=self._sem,
                on_event=self._on_event_factory(session_id, callback_client),
            )
            self._sessions[session_id] = worker
            self._subscribers[session_id] = []

            await worker.start()
            return session_id

    async def stop(self, session_id: str) -> bool:
        async with self._lock:
            worker = self._sessions.get(session_id)
            if not worker:
                return False
            logging.getLogger("stt").info("session %s stop", session_id)
            await worker.stop()
            self._sessions.pop(session_id, None)
            self._subscribers.pop(session_id, None)
            return True

    def list(self) -> Dict[str, Any]:
        return {
            "count": len(self._sessions),
            "sessions": [
                {"session_id": sid, "live_id": w.live_id, "running": w.running}
                for sid, w in self._sessions.items()
            ],
        }

    def subscribe(self, session_id: str) -> Optional[asyncio.Queue]:
        if session_id not in self._sessions:
            return None
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        self._subscribers[session_id].append(q)
        return q

    def unsubscribe(self, session_id: str, q: asyncio.Queue) -> None:
        arr = self._subscribers.get(session_id)
        if not arr:
            return
        try:
            arr.remove(q)
        except ValueError:
            pass

    def _on_event_factory(self, session_id: str, callback_client: Optional[CallbackClient]):
        async def on_event(evt: CaptionEvent):
            msg = evt.model_dump()
            for q in list(self._subscribers.get(session_id, [])):
                try:
                    q.put_nowait(msg)
                except asyncio.QueueFull:
                    pass

            if callback_client:
                payload = {
                    "type": evt.type,
                    "session_id": evt.session_id,
                    "room_id": evt.room_id,
                    "publisher_id": evt.publisher_id,
                    "utterance_id": evt.utterance_id,
                    "seq": evt.seq,
                    "committed_until": evt.committed_until,
                    "segments": [],
                }
                if evt.type in ("partial", "final"):
                    payload["segments"] = [
                        {
                            "start_ms": int(evt.t_start * 1000),
                            "end_ms": int(evt.t_end * 1000),
                            "text": evt.text,
                            "is_final": evt.type == "final",
                            "replace": evt.replace,
                            "stability_hits": evt.stability_hits,
                        }
                    ]
                await callback_client.post(payload)

        return on_event
