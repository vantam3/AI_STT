import asyncio
import logging
import os
import random
import socket
import time
import uuid
from typing import Dict, Optional, Any, Literal, List
from pydantic import BaseModel, Field

from app.stt.worker import STTWorker, CaptionEvent
from app.stt.callbacks import CallbackClient


class InputJRTPOpus(BaseModel):
    type: Literal["janus_rtp_opus"] = "janus_rtp_opus"
    listen_ip: str = "0.0.0.0"
    audio_port: Optional[int] = None
    payload_type: Optional[int] = None
    ssrc: Optional[int] = None


class InputFFmpegURL(BaseModel):
    type: Literal["ffmpeg_url"] = "ffmpeg_url"
    url: str


InputConfig = InputJRTPOpus | InputFFmpegURL


class ASRConfig(BaseModel):
    model_size: Optional[str] = "small"
    language: Optional[str] = "vi"
    vad_filter: bool = False
    beam_size: int = 2
    window_seconds: float = 1.2
    overlap_seconds: float = 0.6
    emit_interval_ms: int = 300
    agreement_hits: int = 2
    silence_seconds: float = 1.0


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
        self._callback_clients: Dict[str, CallbackClient] = {}

        self._sem = asyncio.Semaphore(max_sessions)
        self._callback_queue: asyncio.Queue = asyncio.Queue(maxsize=max_sessions * 400)
        self._callback_task: Optional[asyncio.Task] = None
        self._cb_dropped_partial = 0
        self._cb_dropped_total = 0
        self._cb_sent_total = 0
        self._cb_fail_total = 0
        self._cb_last_log_ts = 0.0
        self._stopping_sessions: set[str] = set()
        self._transcripts: Dict[str, Dict[str, Any]] = {}

    async def _ensure_callback_worker(self) -> None:
        if self._callback_task and not self._callback_task.done():
            return
        self._callback_task = asyncio.create_task(self._callback_worker())

    async def _callback_worker(self) -> None:
        logger = logging.getLogger("stt")
        while True:
            try:
                session_id, client, payload = await self._callback_queue.get()
            except asyncio.CancelledError:
                break
            if payload is None:
                try:
                    await client.close()
                except Exception:
                    logger.exception("callback worker failed to close client")
                continue
            if session_id in self._stopping_sessions:
                self._cb_dropped_total += 1
                if payload.get("type") == "partial":
                    self._cb_dropped_partial += 1
                continue
            try:
                ok = await client.post(payload)
                self._cb_sent_total += 1
                if not ok:
                    self._cb_fail_total += 1
            except Exception:
                logger.exception("callback worker failed")
            now = time.time()
            if now - self._cb_last_log_ts >= 5.0:
                fail_rate = (
                    self._cb_fail_total / self._cb_sent_total
                    if self._cb_sent_total > 0
                    else 0.0
                )
                logger.info(
                    "callback queue qsize=%s dropped_partial=%s dropped_total=%s fail_rate=%.1f%%",
                    self._callback_queue.qsize(),
                    self._cb_dropped_partial,
                    self._cb_dropped_total,
                    fail_rate * 100.0,
                )
                self._cb_last_log_ts = now
        logger.info("callback worker stopped")

    def _allocate_rtp_port(self, listen_ip: str, port_min: int, port_max: int) -> int:
        if port_min < 1 or port_max > 65535 or port_min > port_max:
            raise ValueError("RTP_PORT_MIN/MAX must be within 1-65535 and MIN <= MAX")
        used_ports = {
            w.input_cfg.audio_port
            for w in self._sessions.values()
            if getattr(w.input_cfg, "type", None) == "janus_rtp_opus"
        }
        for _ in range(80):
            candidate = random.randint(port_min, port_max)
            if candidate in used_ports:
                continue
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    sock.bind((listen_ip, candidate))
            except OSError:
                continue
            return candidate
        raise ValueError("no free RTP port available in configured range")

    def _validate_rtp_port(self, listen_ip: str, port: int, port_min: Optional[int], port_max: Optional[int]) -> None:
        if port is None or port <= 0:
            raise ValueError("audio_port must be a positive integer")
        if port_min is not None and port_max is not None:
            if port < port_min or port > port_max:
                raise ValueError(f"audio_port must be within [{port_min}, {port_max}]")
        used_ports = {
            w.input_cfg.audio_port
            for w in self._sessions.values()
            if getattr(w.input_cfg, "type", None) == "janus_rtp_opus"
        }
        if port in used_ports:
            raise ValueError("audio_port already in use by another session")
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind((listen_ip, port))
        except OSError:
            raise ValueError("audio_port not available on host")

    async def start(self, req: StartSessionRequest, rtp_port_range: Optional[tuple[int, int]] = None) -> str:
        async with self._lock:
            if len(self._sessions) >= self.max_sessions:
                raise ValueError(f"too many active sessions (max {self.max_sessions})")

            if req.input.type == "janus_rtp_opus":
                if "payload_type" not in req.input.model_fields_set or req.input.payload_type is None:
                    logging.getLogger("stt").warning(
                        "payload_type missing; defaulting to 111 for janus_rtp_opus"
                    )
                    req.input.payload_type = 111
                if req.input.audio_port is None or req.input.audio_port <= 0:
                    if not rtp_port_range:
                        raise ValueError("audio_port must be provided for janus_rtp_opus")
                    req.input.audio_port = self._allocate_rtp_port(
                        req.input.listen_ip, rtp_port_range[0], rtp_port_range[1]
                    )
                else:
                    pmin = rtp_port_range[0] if rtp_port_range else None
                    pmax = rtp_port_range[1] if rtp_port_range else None
                    self._validate_rtp_port(req.input.listen_ip, req.input.audio_port, pmin, pmax)
                if req.input.payload_type is None or req.input.payload_type <= 0:
                    raise ValueError("payload_type must be a positive integer")

            session_id = uuid.uuid4().hex
            self._stopping_sessions.discard(session_id)
            model_size = req.asr.model_size or self.default_model_size

            backend_url = os.getenv("BACKEND_URL")
            if backend_url:
                backend_url = backend_url.rstrip("/")
            default_callback_url = f"{backend_url}/janus/api/ai/captions" if backend_url else None
            callback_url = req.callback.url or default_callback_url
            callback_secret = os.getenv("AI_KEY")
            callback_client = CallbackClient(callback_url, callback_secret) if callback_url else None
            if callback_client:
                await self._ensure_callback_worker()
                self._callback_clients[session_id] = callback_client

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
                language=req.asr.language or "vi",
                vad_filter=req.asr.vad_filter,
                beam_size=req.asr.beam_size,
                window_seconds=req.asr.window_seconds,
                overlap_seconds=req.asr.overlap_seconds,
                emit_interval_ms=req.asr.emit_interval_ms,
                agreement_hits=req.asr.agreement_hits,
                silence_seconds=req.asr.silence_seconds,
                device=self.device,
                compute_type=self.compute_type,
                sem=self._sem,
                on_event=self._on_event_factory(session_id, callback_client),
            )
            self._sessions[session_id] = worker
            self._subscribers[session_id] = []
            self._transcripts[session_id] = {"text": "", "segments": [], "ended": False}

            await worker.start()
            return session_id

    async def stop(self, session_id: str) -> bool:
        async with self._lock:
            worker = self._sessions.get(session_id)
            if not worker:
                return False
            self._stopping_sessions.add(session_id)
            logging.getLogger("stt").info("session %s stop", session_id)
            await worker.stop()
            self._sessions.pop(session_id, None)
            self._subscribers.pop(session_id, None)
            transcript = self._transcripts.get(session_id)
            if transcript:
                transcript["ended"] = True
            cb = self._callback_clients.pop(session_id, None)
            if cb:
                await self._drop_callback_payloads(session_id)
                if self._callback_task and not self._callback_task.done():
                    try:
                        self._callback_queue.put_nowait((session_id, cb, None))
                    except asyncio.QueueFull:
                        try:
                            await cb.close()
                        except Exception:
                            logging.getLogger("stt").exception(
                                "failed to close callback client session_id=%s", session_id
                            )
                else:
                    try:
                        await cb.close()
                    except Exception:
                        logging.getLogger("stt").exception(
                            "failed to close callback client session_id=%s", session_id
                        )
            if not self._callback_clients and self._callback_task and not self._callback_task.done():
                self._callback_task.cancel()
                try:
                    await self._callback_task
                except asyncio.CancelledError:
                    pass
            return True

    def list(self) -> Dict[str, Any]:
        return {
            "count": len(self._sessions),
            "sessions": [
                {"session_id": sid, "live_id": w.live_id, "running": w.running}
                for sid, w in self._sessions.items()
            ],
        }

    def get_transcript(self, session_id: str) -> Optional[Dict[str, Any]]:
        transcript = self._transcripts.get(session_id)
        if not transcript:
            return None
        return {
            "session_id": session_id,
            "text": transcript["text"],
            "segments": list(transcript["segments"]),
            "ended": bool(transcript.get("ended")),
        }

    async def shutdown(self) -> None:
        async with self._lock:
            for sid, worker in list(self._sessions.items()):
                try:
                    await worker.stop()
                except Exception:
                    logging.getLogger("stt").exception("failed to stop worker on shutdown sid=%s", sid)
            self._sessions.clear()
            self._subscribers.clear()
            self._transcripts.clear()
            for cb in list(self._callback_clients.values()):
                try:
                    await cb.close()
                except Exception:
                    logging.getLogger("stt").exception("failed to close callback client on shutdown")
            self._callback_clients.clear()
            self._stopping_sessions.clear()
            if self._callback_task and not self._callback_task.done():
                self._callback_task.cancel()
                try:
                    await self._callback_task
                except asyncio.CancelledError:
                    pass
            self._callback_task = None

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

    def _append_transcript(self, session_id: str, evt: CaptionEvent) -> None:
        transcript = self._transcripts.get(session_id)
        if not transcript:
            return
        text = evt.text or ""
        if not text:
            return
        transcript["text"] += text
        transcript["segments"].append(
            {
                "seq": evt.seq,
                "utterance_id": evt.utterance_id,
                "start_ms": int(evt.t_start * 1000),
                "end_ms": int(evt.t_end * 1000),
                "text": text,
            }
        )

    def _on_event_factory(self, session_id: str, callback_client: Optional[CallbackClient]):
        async def on_event(evt: CaptionEvent):
            if evt.type == "commit":
                self._append_transcript(session_id, evt)
            msg = evt.model_dump()
            for q in list(self._subscribers.get(session_id, [])):
                try:
                    q.put_nowait(msg)
                except asyncio.QueueFull:
                    pass

            if not callback_client:
                return
            if session_id in self._stopping_sessions:
                return

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
            if evt.type in ("partial", "commit", "final"):
                payload["segments"] = [
                    {
                        "start_ms": int(evt.t_start * 1000),
                        "end_ms": int(evt.t_end * 1000),
                        "text": evt.text,
                        "is_final": evt.type == "final",
                        "replace": evt.replace,
                        "stability_hits": evt.stability_hits,
                        "end_of_turn": evt.end_of_turn,
                    }
                ]

            qsize = self._callback_queue.qsize()
            qmax = self._callback_queue.maxsize
            over_80 = qmax > 0 and (qsize / qmax) >= 0.8
            is_partial = payload.get("type") == "partial"
            if is_partial and (over_80 or self._callback_queue.full()):
                self._cb_dropped_partial += 1
                self._cb_dropped_total += 1
                return
            if self._callback_queue.full() and not is_partial:
                while self._callback_queue.full():
                    try:
                        _, _, dropped = self._callback_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    self._cb_dropped_total += 1
                    if dropped.get("type") == "partial":
                        self._cb_dropped_partial += 1
            try:
                self._callback_queue.put_nowait((session_id, callback_client, payload))
            except asyncio.QueueFull:
                self._cb_dropped_total += 1
                if is_partial:
                    self._cb_dropped_partial += 1
                logging.getLogger("stt").warning(
                    "callback queue full session_id=%s seq=%s",
                    evt.session_id,
                    evt.seq,
                )

        return on_event

    async def _drop_callback_payloads(self, session_id: str) -> None:
        if self._callback_queue.empty():
            return
        items: list[tuple[str, CallbackClient, dict | None]] = []
        try:
            while True:
                item = self._callback_queue.get_nowait()
                if item[0] != session_id:
                    items.append(item)
                else:
                    self._cb_dropped_total += 1
                    if item[2] and item[2].get("type") == "partial":
                        self._cb_dropped_partial += 1
        except asyncio.QueueEmpty:
            pass
        self._callback_queue = asyncio.Queue(maxsize=self._callback_queue.maxsize)
        for item in items:
            try:
                self._callback_queue.put_nowait(item)
            except asyncio.QueueFull:
                self._cb_dropped_total += 1
