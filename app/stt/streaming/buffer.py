class AudioBuffer:
    def __init__(self, sample_rate: int, max_seconds: float):
        self.sample_rate = sample_rate
        self.bytes_per_sec = sample_rate * 2
        self.max_bytes = int(max_seconds * self.bytes_per_sec)
        self._buf = bytearray()
        self._start_sample = 0
        self._total_samples = 0

    def append(self, pcm: bytes) -> None:
        if not pcm:
            return
        self._buf.extend(pcm)
        self._total_samples += len(pcm) // 2
        if len(self._buf) > self.max_bytes:
            trim = len(self._buf) - self.max_bytes
            self._buf = self._buf[trim:]
            self._start_sample += trim // 2

    def append_pcm16(self, pcm: bytes) -> None:
        self.append(pcm)

    def total_samples(self) -> int:
        return self._total_samples

    def start_sample(self) -> int:
        return self._start_sample

    def ready(self, end_sample: int, step_samples: int) -> bool:
        return self._total_samples - end_sample >= step_samples

    def read_window(self, end_sample: int, window_samples: int) -> tuple[bytes, int] | None:
        start_sample = max(self._start_sample, end_sample - window_samples)
        if end_sample <= start_sample:
            return None
        start_offset = max(0, start_sample - self._start_sample)
        end_offset = max(0, end_sample - self._start_sample)
        start_byte = start_offset * 2
        end_byte = end_offset * 2
        if end_byte > len(self._buf):
            return None
        return bytes(self._buf[start_byte:end_byte]), start_sample

    def trim_to_last_seconds(self, keep_seconds: float) -> None:
        keep_seconds = max(0.0, keep_seconds)
        keep_bytes = int(keep_seconds * self.bytes_per_sec)
        if keep_bytes <= 0:
            self._start_sample += len(self._buf) // 2
            self._buf = bytearray()
            return
        if len(self._buf) <= keep_bytes:
            return
        drop_bytes = len(self._buf) - keep_bytes
        self._buf = self._buf[drop_bytes:]
        self._start_sample += drop_bytes // 2
