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


class CaptionEvent(BaseModel):
    type: str  # "partial" | "commit" | "final" | "end"
    live_id: str
    session_id: str
    utterance_id: int
    text: str
    t_start: float
    t_end: float
    committed_until: Optional[float] = None
    stability_hits: Optional[int] = None
    replace: bool = False
    end_of_turn: bool = False
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
        window_seconds: float,
        overlap_seconds: float,
        emit_interval_ms: int,
        agreement_hits: int,
        silence_seconds: float,
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

  
        self.beam_size = int(os.getenv("STT_BEAM_SIZE", str(beam_size or 1)) or "1")


        self.window_seconds = window_seconds
        self.overlap_seconds = overlap_seconds
        self.emit_interval_ms = emit_interval_ms
        self.agreement_hits = agreement_hits
        self.silence_seconds = silence_seconds

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

        self._sr = 16000
        self._bytes_per_sec = self._sr * 2  # mono int16

        max_seconds = float(os.getenv("STT_BUFFER_SECONDS", "8.0") or "8.0")
        buffer_capacity = max_seconds * 5.0
        self._buffer = AudioBuffer(sample_rate=self._sr, max_seconds=buffer_capacity)

        self._gate = SpeechGate(
            rms_threshold=float(os.getenv("STT_RMS_THRESHOLD", "0.005") or "0.005"),
            min_speech_seconds=float(os.getenv("STT_MIN_SPEECH_SECONDS", "0.2") or "0.2"),
            min_speech_ratio=float(os.getenv("STT_MIN_SPEECH_RATIO", "0.15") or "0.15"),
        )

        stable_min_chars = int(os.getenv("STT_STABLE_MIN_CHARS", "6") or "6")
        self._stabilizer = SimpleStabilizer(min_chars=stable_min_chars)
        self._dedupe = TextDedupe()
        self._agreement = LocalAgreement(min_hits=agreement_hits)
        self._asr = FasterWhisperASR(
            self.model_size, self.device, self.compute_type, self.language or "vi"
        )

        self._seq = 0
        self._last_t0 = 0.0
        self._last_t1 = 0.0
        self._last_emit_t1 = -1.0

        self._last_voice_ts = time.time()
        self._ingest_bytes = 0
        self._ingest_start = time.time()
        self._partial_char_limit = int(os.getenv("STT_PARTIAL_CHAR_LIMIT", "220") or "220")

        self._utterance_id = 1
        self._current_committed_text = ""
        self._final_emitted_text = ""
        self._last_display_text = ""
        self._end_sample = 0

        self._last_rtf = 0.0
        self._last_proc_ms = 0.0


        self._window_seconds = self._coalesce_window_seconds()
        self._emit_interval_ms = self._coalesce_emit_interval_ms()
        self._overlap_seconds = self._coalesce_overlap_seconds(self._window_seconds, self._emit_interval_ms)
        self._silence_seconds = self._coalesce_silence_seconds()

        self._lag_drop_seconds = float(os.getenv("STT_LAG_DROP_SECONDS", "3.0") or "3.0")
        self._lag_keep_seconds = float(os.getenv("STT_LAG_KEEP_SECONDS", "1.5") or "1.5")

        self._skip_windows = 0
        self._last_ff_sample = 0
        self._last_ff_ts = 0.0
        self._playout_start = time.time()
        self._playout_ref_sample = 0
        self._prompt_cooldown = 0


        self._dump_wave: Optional[wave.Wave_write] = None
        self._dump_path = ""


        self._buf_lock = asyncio.Lock()

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
                await asyncio.wait_for(self._task, timeout=1.0)
            except asyncio.TimeoutError:
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
            except asyncio.CancelledError:
                pass
        self._task = None

    async def _emit(
        self,
        typ: str,
        text: str,
        t0: float,
        t1: float,
        replace: bool = False,
        end_of_turn: bool = False,
    ):
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
            committed_until=None,
            stability_hits=self._agreement.hits() if typ in ("partial", "final") else None,
            replace=replace,
            end_of_turn=end_of_turn,
            speaker=None,
            room_id=self.room_id,
            publisher_id=self.publisher_id,
            seq=self._seq,
        )
        await self.on_event(evt)

    @staticmethod
    def _common_prefix_len(a: str, b: str) -> int:
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return i

    def _build_prompt(self) -> Optional[str]:
        prompt_chars = int(os.getenv("STT_PROMPT_CHARS", "240") or "240")
        if self._prompt_cooldown > 0:
            return None
        txt = (self._current_committed_text or self._final_emitted_text or "").strip()
        if not txt:
            return None
        return txt[-prompt_chars:] if len(txt) > prompt_chars else txt

    def _coalesce_window_seconds(self) -> float:
        v = os.getenv("STT_WINDOW_SECONDS")
        if v:
            try:
                return float(v)
            except Exception:
                pass
        if self.window_seconds and self.window_seconds > 0:
            return float(self.window_seconds)
        return 4.0

    def _coalesce_emit_interval_ms(self) -> int:
        v = os.getenv("STT_EMIT_INTERVAL_MS")
        if v:
            try:
                return int(float(v))
            except Exception:
                pass
        if self.emit_interval_ms and self.emit_interval_ms > 0:
            return int(self.emit_interval_ms)
        return 400

    def _coalesce_overlap_seconds(self, window_seconds: float, emit_interval_ms: int) -> float:
        v = os.getenv("STT_OVERLAP_SECONDS")
        if v:
            try:
                ov = float(v)
                return min(max(0.0, ov), max(0.0, window_seconds - 0.05))
            except Exception:
                pass
        if self.overlap_seconds and self.overlap_seconds > 0:
            return min(float(self.overlap_seconds), max(0.0, window_seconds - 0.05))
        return min(1.0, max(0.0, window_seconds - 0.05))

    def _coalesce_silence_seconds(self) -> float:
        v = os.getenv("STT_SILENCE_SECONDS")
        if v:
            try:
                return float(v)
            except Exception:
                pass
        if self.silence_seconds and self.silence_seconds > 0:
            return float(self.silence_seconds)
        return 0.8

    def _fast_forward_cursor(self, target_end_sample: int, reason: str, logger, extra: str = "") -> None:
        if target_end_sample <= self._end_sample:
            return
        now = time.time()
        if now - self._last_ff_ts < 1.0 and target_end_sample - self._last_ff_sample < self._sr * 0.2:
            return
        self._end_sample = target_end_sample
        self._skip_windows = max(self._skip_windows, 2)
        self._last_ff_sample = target_end_sample
        self._last_ff_ts = now
        logger.warning(
            "session %s %s fast-forward cursor to %.2fs%s",
            self.session_id,
            reason,
            self._end_sample / self._sr,
            f" {extra}" if extra else "",
        )

    async def _flush_final_once(self) -> None:
        if self._final_flushed:
            return
        self._final_flushed = True

        final_text = (self._last_display_text or "").strip()
        if not final_text or final_text == self._final_emitted_text:
            return

        if final_text != self._current_committed_text:
            new_text = final_text
            if final_text.startswith(self._current_committed_text):
                new_text = final_text[len(self._current_committed_text):]
            new_text = new_text.lstrip(" \t\n?!.…,-:;")
            if new_text:
                await self._emit("commit", new_text, self._last_t0, self._last_t1, replace=False)

        await self._emit("final", final_text, self._last_t0, self._last_t1, end_of_turn=True)

        self._final_emitted_text = final_text
        self._current_committed_text = final_text
        self._last_display_text = final_text
        self._utterance_id += 1

    def _compute_window(
        self,
        total_samples: int,
        chunk_samples: int,
        step_samples: int,
        max_end_sample: Optional[int] = None,
    ) -> Optional[tuple[bytes, int, float]]:
        if chunk_samples <= 0 or step_samples <= 0:
            return None
        limit = max_end_sample if max_end_sample is not None else total_samples
        if self._end_sample == 0:
            if limit < chunk_samples:
                return None
            self._end_sample = chunk_samples
        elif not self._buffer.ready(self._end_sample, step_samples):
            return None
        else:
            self._end_sample += step_samples
        if self._end_sample > limit:
            return None
        win = self._buffer.read_window(self._end_sample, chunk_samples)
        if win is None:
            return None
        audio_bytes, win_start_sample = win
        t1 = self._end_sample / self._sr
        return audio_bytes, win_start_sample, t1

    def _update_playout(self, now: float) -> float:
        elapsed = max(0.0, now - self._playout_start)
        playout_sample = self._playout_ref_sample + int(elapsed * self._sr)
        return playout_sample / self._sr

    async def _run(self):
        logger = logging.getLogger("stt")
        self._playout_start = time.time()
        self._playout_ref_sample = 0

        async def _ingest_loop():
            if os.getenv("STT_DUMP_FULL", "0") == "1":
                self._dump_path = f"stt_dump_{self.session_id}.wav"
                self._dump_wave = wave.open(self._dump_path, "wb")
                self._dump_wave.setnchannels(1)
                self._dump_wave.setsampwidth(2)
                self._dump_wave.setframerate(self._sr)
                logger.info("session %s dump FULL audio to %s", self.session_id, self._dump_path)

            async with self.sem:
                async for pcm_bytes in pcm_stream_from_ffmpeg(self.input_cfg, stop_event=self._stop_event):
                    if self._stop_event.is_set() or self._stop_requested:
                        break
                    if not pcm_bytes:
                        continue

                    async with self._buf_lock:
                        self._buffer.append_pcm16(pcm_bytes)

                    self._ingest_bytes += len(pcm_bytes)

                    if self._dump_wave is not None:
                        self._dump_wave.writeframes(pcm_bytes)

        ingest_task = asyncio.create_task(_ingest_loop())

        try:
            while True:
                if self._stop_event.is_set() or self._stop_requested:
                    break

                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=self._emit_interval_ms / 1000.0)
                    break
                except asyncio.TimeoutError:
                    pass

                now = time.time()
                async with self._buf_lock:
                    total_samples = self._buffer.total_samples()

                playout_t = self._update_playout(now)
                raw_latest_t = total_samples / self._sr
                lag_sec = raw_latest_t - playout_t

                if lag_sec > self._lag_drop_seconds:
                    target = int(max(0.0, (raw_latest_t - self._lag_keep_seconds)) * self._sr)
                    ingest_elapsed = max(1e-6, now - self._ingest_start)
                    ingest_rtf = (self._ingest_bytes / self._bytes_per_sec) / ingest_elapsed
                    self._fast_forward_cursor(
                        target_end_sample=target,
                        reason="burst ingest_rtf=%.2f" % ingest_rtf,
                        logger=logger,
                        extra="reset playout to %.2fs (keep=%.2fs)"
                        % (raw_latest_t - self._lag_keep_seconds, self._lag_keep_seconds),
                    )
                    self._playout_start = now
                    self._playout_ref_sample = target

                if self._skip_windows > 0:
                    self._skip_windows -= 1
                    continue

                window_seconds = self._window_seconds
                step_seconds = max(0.10, window_seconds - self._overlap_seconds)

                chunk_samples = int(window_seconds * self._sr)
                step_samples = max(1, int(step_seconds * self._sr))

                async with self._buf_lock:
                    win = self._compute_window(total_samples, chunk_samples, step_samples)

                if win is None:
                    continue

                audio_bytes, win_start_sample, t1 = win
                win_sec = len(audio_bytes) / self._bytes_per_sec

                min_win = float(os.getenv("STT_MIN_WINDOW_SECONDS", "2.0") or "2.0")
                if win_sec < min_win:
                    continue

                audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                if not self._gate.should_transcribe(audio):
                    continue

                t0 = win_start_sample / self._sr
                self._last_t0 = t0
                self._last_t1 = t1

                prompt = self._build_prompt()
                if self._prompt_cooldown > 0:
                    self._prompt_cooldown -= 1

                start_ts = time.time()
                segments, info = await asyncio.to_thread(
                    self._asr.transcribe, audio, self.vad_filter, self.beam_size, prompt
                )
                proc_s = time.time() - start_ts
                self._last_rtf = (proc_s / max(1e-6, win_sec))
                self._last_proc_ms = proc_s * 1000.0

                seg_texts = []
                speech_seconds = 0.0
                for seg in segments:
                    seg_texts.append(seg.text)
                    try:
                        speech_seconds += max(0.0, float(seg.end) - float(seg.start))
                    except Exception:
                        pass

                text = "".join(seg_texts).strip()

                if speech_seconds > 0.0:
                    self._last_voice_ts = now

                logger.info(
                    "session %s rtf=%.2f proc=%.3fs audio=%.3fs",
                    self.session_id,
                    self._last_rtf,
                    proc_s,
                    win_sec,
                )

                if not self._gate.should_emit(speech_seconds, win_sec):
                    continue
                if not text:
                    continue

                st = self._stabilizer.update(text)
                full_text = (
                    self._current_committed_text
                    + (" " if self._current_committed_text and st.partial else "")
                    + st.partial
                ).strip()

                if len(full_text) > self._partial_char_limit:
                    full_text = full_text[: self._partial_char_limit]

                prefix_len = self._common_prefix_len(self._current_committed_text, full_text)
                merged_partial = full_text[prefix_len:].lstrip()
                if not merged_partial:
                    continue

                if self._dedupe.is_repetitive(merged_partial):
                    logger.warning("session %s repetitive output detected; resetting prompt", self.session_id)
                    self._prompt_cooldown = max(self._prompt_cooldown, 3)
                    self._stabilizer.reset()
                    self._agreement.update("")
                    self._dedupe.update("")
                    continue
                if self._dedupe.is_duplicate(merged_partial):
                    continue

                if abs(t1 - self._last_emit_t1) < 1e-6:
                    continue

                self._last_emit_t1 = t1

                self._agreement.update(st.stable)
                await self._emit("partial", full_text, t0, t1, replace=True)
                self._last_display_text = full_text
                self._dedupe.update(merged_partial)

                if self._agreement.should_emit_final(st.stable) and st.stable != self._current_committed_text:
                    new_text = st.stable
                    if st.stable.startswith(self._current_committed_text):
                        new_text = st.stable[len(self._current_committed_text):]
                    new_text = new_text.lstrip(" \t\n?!.…,-:;")
                    if new_text:
                        await self._emit("commit", new_text, t0, t1, replace=False)

                    self._current_committed_text = st.stable

                if (now - self._last_voice_ts) >= self._silence_seconds and full_text and full_text != self._final_emitted_text:
                    final_text = full_text.strip()

                    if final_text != self._current_committed_text:
                        new_text = final_text
                        if final_text.startswith(self._current_committed_text):
                            new_text = final_text[len(self._current_committed_text):]
                        new_text = new_text.lstrip(" \t\n?!.…,-:;")
                        if new_text:
                            await self._emit("commit", new_text, t0, t1, replace=False)

                    await self._emit("final", final_text, t0, t1, end_of_turn=True)

                    self._final_emitted_text = final_text
                    self._current_committed_text = final_text
                    self._last_display_text = final_text
                    self._utterance_id += 1

                    self._stabilizer.reset()
                    self._agreement.update("")
                    self._dedupe.update("")

        except Exception:
            logger.exception("session %s worker failed", self.session_id)
        finally:
            if not ingest_task.done():
                ingest_task.cancel()
                try:
                    await ingest_task
                except asyncio.CancelledError:
                    pass

            if not self._end_emitted:
                await self._flush_final_once()
                await self._emit("end", "", self._last_t0, self._last_t1)
                self._end_emitted = True

            self.running = False
            if self._dump_wave is not None:
                self._dump_wave.close()
            logger.info("session %s end", self.session_id)
