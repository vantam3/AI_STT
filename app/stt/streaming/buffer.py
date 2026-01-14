class AudioBuffer:
    def __init__(self, sample_rate: int, max_seconds: float):
        self.sample_rate = sample_rate
        self.bytes_per_sec = sample_rate * 2
        self.max_bytes = int(max_seconds * self.bytes_per_sec)
        self.max_seconds = max_seconds
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

    def total_samples(self) -> int:
        return self._total_samples

    def ready(self, end_sample: int, chunk_seconds: float) -> bool:
        need = int(chunk_seconds * self.sample_rate)
        return self._total_samples - end_sample >= need

    def window_for_end(self, end_sample: int) -> bytes:
        window_samples = int(self.max_seconds * self.sample_rate)
        start_sample = max(self._start_sample, end_sample - window_samples)
        start_offset = max(0, start_sample - self._start_sample)
        end_offset = max(0, end_sample - self._start_sample)
        start_byte = start_offset * 2
        end_byte = end_offset * 2
        return bytes(self._buf[start_byte:end_byte])

    def trim_to_time(self, end_time: float, keep_seconds: float) -> None:
        target_time = max(0.0, end_time - keep_seconds)
        target_sample = int(target_time * self.sample_rate)
        if target_sample <= self._start_sample:
            return
        drop_samples = min(target_sample - self._start_sample, len(self._buf) // 2)
        if drop_samples <= 0:
            return
        drop_bytes = drop_samples * 2
        self._buf = self._buf[drop_bytes:]
        self._start_sample += drop_samples
