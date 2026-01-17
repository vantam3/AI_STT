import logging
import os
import threading
from typing import Dict, Optional, Tuple

from faster_whisper import WhisperModel


class FasterWhisperASR:
    _models: Dict[Tuple[str, str, str], WhisperModel] = {}
    _lock = threading.Lock()

    def __init__(self, model_size: str, device: str, compute_type: str, language: str):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language

    def _load_model(self) -> WhisperModel:
        key = (self.model_size, self.device, self.compute_type)
        with self.__class__._lock:
            model = self.__class__._models.get(key)
            if model is None:
                logging.getLogger("stt").info("loading model=%s device=%s compute_type=%s", *key)

                cpu_threads = int(os.getenv("STT_CPU_THREADS", "0") or "0")
                num_workers = int(os.getenv("STT_NUM_WORKERS", "1") or "1")

                kwargs = {}
                if cpu_threads > 0:
                    kwargs["cpu_threads"] = cpu_threads
                if num_workers > 0:
                    kwargs["num_workers"] = num_workers

                model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                    **kwargs,
                )
                self.__class__._models[key] = model
                logging.getLogger("stt").info("model loaded key=%s threads=%s workers=%s", key, cpu_threads, num_workers)
            return model

    def transcribe(self, audio, vad_filter: bool, beam_size: int, prompt: Optional[str] = None):
        model = self._load_model()

        no_speech_threshold = float(os.getenv("STT_NO_SPEECH_THRESHOLD", "0.70") or "0.70")
        log_prob_threshold = float(os.getenv("STT_LOGPROB_THRESHOLD", "-1.0") or "-1.0")
        compression_ratio_threshold = float(os.getenv("STT_COMPRESSION_RATIO_THRESHOLD", "2.4") or "2.4")

        condition_on_previous_text = os.getenv("STT_CONDITION_ON_PREV_TEXT", "1") != "0"

        return model.transcribe(
            audio,
            language=self.language,
            vad_filter=vad_filter,
            beam_size=int(beam_size or 1),
            condition_on_previous_text=condition_on_previous_text,
            initial_prompt=prompt,
            temperature=0.0,
            no_speech_threshold=no_speech_threshold,
            log_prob_threshold=log_prob_threshold,
            compression_ratio_threshold=compression_ratio_threshold,
        )
