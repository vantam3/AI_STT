import logging
from typing import Optional

from faster_whisper import WhisperModel


class FasterWhisperASR:
    def __init__(self, model_size: str, device: str, compute_type: str, language: str):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self._model: Optional[WhisperModel] = None

    def _load_model(self) -> WhisperModel:
        if self._model is None:
            logging.getLogger("stt").info("loading model=%s", self.model_size)
            self._model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
            logging.getLogger("stt").info("model loaded")
        return self._model

    def transcribe(self, audio, vad_filter: bool, beam_size: int):
        model = self._load_model()
        return model.transcribe(
            audio,
            language=self.language,
            vad_filter=vad_filter,
            beam_size=beam_size,
            condition_on_previous_text=True,
            temperature=0.0,
        )
