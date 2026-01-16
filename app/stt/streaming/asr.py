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
                logging.getLogger("stt").info(
                    "loading model=%s device=%s compute_type=%s", *key
                )
                model = WhisperModel(
                    self.model_size, device=self.device, compute_type=self.compute_type
                )
                self.__class__._models[key] = model
                logging.getLogger("stt").info("model loaded key=%s", key)
            return model

    def transcribe(self, audio, vad_filter: bool, beam_size: int, prompt: Optional[str] = None):
        model = self._load_model()

        # Anti-hallucination / streaming-safety knobs (env-driven).
        # These help avoid "loop" outputs like repeating the same phrase.
        # Tune per language/content if needed.
        no_speech_threshold = float(os.getenv("STT_NO_SPEECH_THRESHOLD", "0.60") or "0.60")
        log_prob_threshold = float(os.getenv("STT_LOGPROB_THRESHOLD", "-1.0") or "-1.0")
        compression_ratio_threshold = float(os.getenv("STT_COMPRESSION_RATIO_THRESHOLD", "2.4") or "2.4")

        return model.transcribe(
            audio,
            language=self.language,
            vad_filter=vad_filter,
            beam_size=beam_size,
            condition_on_previous_text=True,
            initial_prompt=prompt,
            temperature=0.0,
            no_speech_threshold=no_speech_threshold,
            log_prob_threshold=log_prob_threshold,
            compression_ratio_threshold=compression_ratio_threshold,
        )
