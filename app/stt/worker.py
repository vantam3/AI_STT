import asyncio
import logging
import os
import time
import wave
from typing import Optional, Callable, Awaitable

import numpy as np
from pydantic import BaseModel

from app.stt.ffmpeg_reader import pcm_stream_from_ffmpeg
from app.stt.stabilizer import SimpleStabilizer
from app.stt.streaming.asr import FasterWhisperASR
from app.stt.streaming.buffer import AudioBuffer
from app.stt.streaming.gate import SpeechGate
from app.stt.streaming.dedupe import TextDedupe
from app.stt.streaming.agreement import LocalAgreement
from app.stt.streaming.jitter import JitterBuffer


class CaptionEvent(BaseModel):
    type: str  # "partial" | "final" | "end"
    live_id: str
    session_id: str
    utterance_id: int
    text: str
    t_start: float
    t_end: float
    committed_until: Optional[float] = None
    stability_hits: Optional[int] = None
    replace: bool = False
    speaker: Optional[str] = None
    room_id: Optional[str] = None
    publisher_id: Optional[str] = None
    seq: int


class STTWorker:
    def __init__(
        self,
        session_id: str,
        live_id: str,
        room_id: Optional[str],
        publisher_id: Optional[str],
        input_cfg,
        model_size: str,
        language: Optional[str],
        vad_filter: bool,
        beam_size: int,
        chunk_seconds: float,
        overlap_seconds: float,
        emit_interval_ms: int,
        device: str,
        compute_type: str,
        sem: asyncio.Semaphore,
        on_event: Callable[[CaptionEvent], Awaitable[None]],
    ):
        self.session_id = session_id
        self.live_id = live_id
        self.room_id = room_id
        self.publisher_id = publisher_id
        self.input_cfg = input_cfg

        self.model_size = model_size
        self.language = language
        self.vad_filter = vad_filter
        self.beam_size = beam_size

        self.chunk_seconds = chunk_seconds
        self.overlap_seconds = overlap_seconds
        self.emit_interval_ms = emit_interval_ms

        self.device = device
        self.compute_type = compute_type
        self.sem = sem
        self.on_event = on_event

        self.running = False
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._stop_requested = False
        self._final_flushed = False
        self._end_emitted = False

        self._stabilizer = SimpleStabilizer(min_chars=18)

        self._sr = 16000
        self._bytes_per_sec = self._sr * 2  # mono int16
        self._buffer = AudioBuffer(sample_rate=self._sr, max_seconds=6.0)
        self._asr = FasterWhisperASR(self.model_size, self.device, self.compute_type, self.language or "vi")
        self._gate = SpeechGate(rms_threshold=0.005, min_speech_seconds=0.2, min_speech_ratio=0.2)
        self._dedupe = TextDedupe()
        self._agreement = LocalAgreement(min_hits=3)
        self._jitter = JitterBuffer(sample_rate=self._sr, frame_ms=20, target_delay_ms=40, max_delay_ms=120)

        self._audio_seconds_seen = 0.0
        self._seq = 0
        self._last_t0 = 0.0
        self._last_t1 = 0.0
        self._bytes_seen = 0
        self._last_bytes_log = 0.0
        self._last_voice_ts = time.time()
        self._last_emit_t1 = -1.0

        self._utterance_id = 1
        self._current_committed_text = ""
        self._committed_end_time = 0.0
        self._final_emitted_text = ""
        self._last_segments = None
        self._keep_seconds = 0.8

        self._dump_bytes_left = 0
        self._dump_wave: Optional[wave.Wave_write] = None
        self._dump_path = ""
        self._text_log_path = ""
        self._text_log = None
        self._asr_dump_bytes_left = 0
        self._asr_dump_wave: Optional[wave.Wave_write] = None
        self._asr_dump_path = ""
        self._end_sample = 0

    async def start(self):
        if self.running:
            return
        self._stop_event.clear()
        self._stop_requested = False
        self._final_flushed = False
        self._end_emitted = False
        self.running = True
        self._task = asyncio.create_task(self._run())

    async def stop(self):
        self._stop_requested = True
        self.running = False
        self._stop_event.set()
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=3.0)
            except asyncio.TimeoutError:
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
            except asyncio.CancelledError:
                pass
        self._task = None

    async def _emit(self, typ: str, text: str, t0: float, t1: float, replace: bool = False):
        self._seq += 1
        logging.getLogger("stt").info(
            "session %s emit %s seq=%s chars=%s text=%s",
            self.session_id,
            typ,
            self._seq,
            len(text),
            text,
        )
        evt = CaptionEvent(
            type=typ,
            live_id=self.live_id,
            session_id=self.session_id,
            utterance_id=self._utterance_id,
            text=text,
            t_start=t0,
            t_end=t1,
            committed_until=self._committed_end_time if typ in ("final", "end") else None,
            stability_hits=self._agreement.hits() if typ in ("partial", "final") else None,
            replace=replace,
            speaker=None,
            room_id=self.room_id,
            publisher_id=self.publisher_id,
            seq=self._seq,
        )
        await self.on_event(evt)
        if self._text_log is not None:
            self._text_log.write(f"{typ}\t{self._seq}\t{self._utterance_id}\t{int(replace)}\t{text}\n")
            self._text_log.flush()

    @staticmethod
    def _common_prefix_len(a: str, b: str) -> int:
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return i

    def _commit(self, stable_text: str, segments, t0: float, keep_seconds: float) -> None:
        if not stable_text:
            return
        prefix_len = self._common_prefix_len(self._current_committed_text, stable_text)
        new_text = stable_text[prefix_len:].lstrip()
        if not new_text:
            return
        target_len = len(stable_text)
        abs_end = None
        acc = 0
        for seg in segments or []:
            seg_text = getattr(seg, "text", "")
            acc += len(seg_text)
            if acc >= target_len:
                try:
                    abs_end = t0 + float(seg.end)
                except Exception:
                    abs_end = None
                break
        if abs_end is None:
            abs_end = t0
        self._current_committed_text = stable_text
        self._committed_end_time = max(self._committed_end_time, abs_end)
        self._buffer.trim_to_time(self._committed_end_time, keep_seconds)

    async def _flush_final_once(self) -> None:
        if self._final_flushed:
            return
        self._final_flushed = True
        st = self._stabilizer.update("")
        if not st.stable:
            return
        if st.stable == self._final_emitted_text:
            return
        await self._emit("final", st.stable, self._last_t0, self._last_t1)
        self._commit(st.stable, self._last_segments, self._last_t0, self._keep_seconds)
        self._final_emitted_text = st.stable
        self._utterance_id += 1
        self._current_committed_text = ""

    async def _run(self):
        async with self.sem:
            logger = logging.getLogger("stt")
            self._bytes_seen = 0
            self._last_bytes_log = time.time()
            dump_full = os.getenv("STT_DUMP_FULL", "0") == "1"
            dump_seconds = float(os.getenv("STT_DUMP_SECONDS", "0") or "0")
            if dump_full or dump_seconds > 0:
                self._dump_bytes_left = -1 if dump_full else int(dump_seconds * self._bytes_per_sec)
                self._dump_path = f"stt_dump_{self.session_id}.wav"
                self._dump_wave = wave.open(self._dump_path, "wb")
                self._dump_wave.setnchannels(1)
                self._dump_wave.setsampwidth(2)
                self._dump_wave.setframerate(self._sr)
                if dump_full:
                    logger.info("session %s dump audio to %s (full)", self.session_id, self._dump_path)
                else:
                    logger.info("session %s dump audio to %s (%ss)", self.session_id, self._dump_path, dump_seconds)
            if os.getenv("STT_DUMP_TEXT", "0") == "1":
                self._text_log_path = f"stt_text_{self.session_id}.txt"
                self._text_log = open(self._text_log_path, "w", encoding="utf-8")
                logger.info("session %s dump text to %s", self.session_id, self._text_log_path)
            asr_dump_full = os.getenv("STT_DUMP_ASR_FULL", "0") == "1"
            asr_dump_seconds = float(os.getenv("STT_DUMP_ASR_SECONDS", "0") or "0")
            if asr_dump_full or asr_dump_seconds > 0:
                self._asr_dump_bytes_left = -1 if asr_dump_full else int(asr_dump_seconds * self._bytes_per_sec)
                self._asr_dump_path = f"stt_asr_window_{self.session_id}.wav"
                self._asr_dump_wave = wave.open(self._asr_dump_path, "wb")
                self._asr_dump_wave.setnchannels(1)
                self._asr_dump_wave.setsampwidth(2)
                self._asr_dump_wave.setframerate(self._sr)
                if asr_dump_full:
                    logger.info("session %s dump asr window to %s (full)", self.session_id, self._asr_dump_path)
                else:
                    logger.info("session %s dump asr window to %s (%ss)", self.session_id, self._asr_dump_path, asr_dump_seconds)
            if self.input_cfg.type == "janus_rtp_opus":
                logger.info(
                    "session %s start: janus_rtp_opus audio_port=%s payload_type=%s room_id=%s publisher_id=%s",
                    self.session_id,
                    self.input_cfg.audio_port,
                    self.input_cfg.payload_type,
                    self.room_id,
                    self.publisher_id,
                )
            else:
                logger.info(
                    "session %s start: ffmpeg_url url=%s room_id=%s publisher_id=%s",
                    self.session_id,
                    self.input_cfg.url,
                    self.room_id,
                    self.publisher_id,
                )

            last_emit = 0.0
            self._keep_seconds = float(os.getenv("STT_COMMIT_KEEP_SECONDS", "0.8") or "0.8")

            try:
                logger.info("session %s start ffmpeg reader", self.session_id)
                async for pcm in pcm_stream_from_ffmpeg(self.input_cfg, stop_event=self._stop_event):
                    if not self.running:
                        break

                    if self._dump_wave is not None and self._dump_bytes_left != 0:
                        if self._dump_bytes_left < 0:
                            self._dump_wave.writeframes(pcm)
                        else:
                            take = min(len(pcm), self._dump_bytes_left)
                            if take > 0:
                                self._dump_wave.writeframes(pcm[:take])
                                self._dump_bytes_left -= take
                                if self._dump_bytes_left == 0:
                                    self._dump_wave.close()
                                    self._dump_wave = None
                                    logger.info("session %s dump done %s", self.session_id, self._dump_path)

                    frames = self._jitter.add(pcm)
                    for frame in frames:
                        self._buffer.append(frame)
                        self._audio_seconds_seen += len(frame) / self._bytes_per_sec
                        self._bytes_seen += len(frame)

                    now_ts = time.time()
                    if now_ts - self._last_bytes_log >= 5.0:
                        kbps = (self._bytes_seen * 8.0) / max(1.0, (now_ts - self._last_bytes_log)) / 1000.0
                        logger.info(
                            "session %s received audio bytes=%s (~%.1f kbps)",
                            self.session_id,
                            self._bytes_seen,
                            kbps,
                        )
                        self._bytes_seen = 0
                        self._last_bytes_log = now_ts

                    now = time.time()
                    if (now - last_emit) * 1000.0 < self.emit_interval_ms:
                        continue

                    if not self._buffer.ready(self._end_sample, self.chunk_seconds):
                        continue

                    chunk_samples = int(self.chunk_seconds * self._sr)
                    end_sample = self._end_sample + chunk_samples
                    if end_sample > self._buffer.total_samples():
                        continue
                    window = self._buffer.window_for_end(end_sample)
                    self._end_sample = end_sample

                    burst_sec = float(os.getenv("STT_BURST_CHECK_SECONDS", "0.5") or "0.5")
                    burst_corr = float(os.getenv("STT_BURST_CORR_THRESHOLD", "0.995") or "0.995")
                    if burst_sec > 0:
                        burst_samples = int(burst_sec * self._sr)
                        if len(window) >= burst_samples * 2 * 2:
                            audio_i16 = np.frombuffer(window, dtype=np.int16)
                            tail = audio_i16[-burst_samples:]
                            prev = audio_i16[-2 * burst_samples:-burst_samples]
                            tail = tail - tail.mean()
                            prev = prev - prev.mean()
                            denom = (np.linalg.norm(tail) * np.linalg.norm(prev)) + 1e-6
                            corr = float(np.dot(tail, prev) / denom)
                            if corr >= burst_corr:
                                logger.info("session %s drop repeated burst corr=%.3f", self.session_id, corr)
                                last_emit = now
                                continue

                    if self._asr_dump_wave is not None and self._asr_dump_bytes_left != 0:
                        if self._asr_dump_bytes_left < 0:
                            self._asr_dump_wave.writeframes(window)
                        else:
                            take = min(len(window), self._asr_dump_bytes_left)
                            if take > 0:
                                self._asr_dump_wave.writeframes(window[:take])
                                self._asr_dump_bytes_left -= take
                                if self._asr_dump_bytes_left == 0:
                                    self._asr_dump_wave.close()
                                    self._asr_dump_wave = None
                                    logger.info("session %s asr window dump done %s", self.session_id, self._asr_dump_path)

                    audio_i16 = np.frombuffer(window, dtype=np.int16)
                    audio = (audio_i16.astype(np.float32) / 32768.0)
                    if not self._gate.should_transcribe(audio):
                        continue

                    t_proc0 = time.time()
                    segments, info = self._asr.transcribe(audio, self.vad_filter, self.beam_size)
                    t_proc = time.time() - t_proc0
                    audio_seconds = max(0.001, len(window) / self._bytes_per_sec)
                    rtf = t_proc / audio_seconds
                    logger.info("session %s rtf=%.2f proc=%.3fs audio=%.3fs", self.session_id, rtf, t_proc, audio_seconds)
                    self._last_segments = segments

                    text = ""
                    speech_seconds = 0.0
                    t0 = max(0.0, self._audio_seconds_seen - (len(window) / self._bytes_per_sec))
                    t1 = self._audio_seconds_seen

                    for seg in segments:
                        text += seg.text
                        try:
                            speech_seconds += max(0.0, float(seg.end) - float(seg.start))
                        except Exception:
                            pass

                    text = text.strip()
                    window_seconds = max(0.001, len(window) / self._bytes_per_sec)
                    if not self._gate.should_emit(speech_seconds, window_seconds):
                        last_emit = now
                        continue
                    if not text:
                        last_emit = now
                        continue

                    st = self._stabilizer.update(text)
                    full_text = (st.stable + (" " if st.stable and st.partial else "") + st.partial).strip()
                    prefix_len = self._common_prefix_len(self._current_committed_text, full_text)
                    merged_partial = full_text[prefix_len:].lstrip()
                    if not merged_partial:
                        last_emit = now
                        continue
                    if self._dedupe.is_repetitive(merged_partial) or self._dedupe.is_duplicate(merged_partial):
                        last_emit = now
                        continue

                    if abs(t1 - self._last_emit_t1) < 1e-6:
                        last_emit = now
                        continue
                    self._last_emit_t1 = t1
                    self._last_t0 = t0
                    self._last_t1 = t1
                    self._last_voice_ts = now
                    self._agreement.update(st.stable)
                    await self._emit("partial", full_text, t0, t1, replace=True)
                    self._dedupe.update(merged_partial)

                    keep_seconds = self._keep_seconds
                    if self._agreement.should_emit_final(st.stable) and st.stable != self._final_emitted_text:
                        await self._emit("final", st.stable, t0, t1)
                        self._commit(st.stable, segments, t0, keep_seconds)
                        self._final_emitted_text = st.stable
                        self._utterance_id += 1
                        self._current_committed_text = ""

                    last_emit = now

                    if (now - self._last_voice_ts) >= 0.7 and st.stable and st.stable != self._final_emitted_text:
                        await self._emit("final", st.stable, t0, t1)
                        self._commit(st.stable, segments, t0, keep_seconds)
                        self._final_emitted_text = st.stable
                        self._utterance_id += 1
                        self._current_committed_text = ""

                if self._stop_requested:
                    await self._flush_final_once()
                    await self._emit("end", "", self._last_t0, self._last_t1)
                    self._end_emitted = True
                else:
                    st = self._stabilizer.update("")
                    if st.stable:
                        await self._emit("final", st.stable, self._last_t0, self._last_t1)
                        self._commit(st.stable, self._last_segments, self._last_t0, self._keep_seconds)
                        self._utterance_id += 1
                        self._current_committed_text = ""
            except Exception:
                return
            finally:
                if self._stop_requested and not self._end_emitted:
                    await self._flush_final_once()
                    await self._emit("end", "", self._last_t0, self._last_t1)
                    self._end_emitted = True
                self.running = False
                if self._dump_wave is not None:
                    self._dump_wave.close()
                if self._text_log is not None:
                    self._text_log.close()
                if self._asr_dump_wave is not None:
                    self._asr_dump_wave.close()
                logger.info("session %s end", self.session_id)
