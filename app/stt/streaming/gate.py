import numpy as np


class SpeechGate:
    def __init__(self, rms_threshold: float, min_speech_seconds: float, min_speech_ratio: float):
        self.rms_threshold = rms_threshold
        self.min_speech_seconds = min_speech_seconds
        self.min_speech_ratio = min_speech_ratio

    def should_transcribe(self, audio: np.ndarray) -> bool:
        if audio.size == 0:
            return False
        rms = float(np.sqrt(np.mean(audio ** 2)))
        return rms >= self.rms_threshold

    def should_emit(self, speech_seconds: float, window_seconds: float) -> bool:
        if speech_seconds < self.min_speech_seconds:
            return False
        if window_seconds <= 0.0:
            return False
        return (speech_seconds / window_seconds) >= self.min_speech_ratio
