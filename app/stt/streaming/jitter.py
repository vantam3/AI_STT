import zlib
from collections import deque
from typing import Deque, List


class JitterBuffer:
    def __init__(self, sample_rate: int, frame_ms: int = 20, target_delay_ms: int = 80, max_delay_ms: int = 240):
        self.sample_rate = sample_rate
        self.frame_bytes = int(sample_rate * 2 * (frame_ms / 1000.0))
        self.target_delay_ms = target_delay_ms
        self.max_delay_ms = max_delay_ms
        self._pending = bytearray()
        self._queue: Deque[bytes] = deque()
        self._hashes: Deque[int] = deque(maxlen=6)
        self._total_frames = 0
        self._dup_dropped = 0
        self._old_dropped = 0

    def add(self, pcm: bytes) -> List[bytes]:
        if not pcm:
            return []
        self._pending.extend(pcm)
        while len(self._pending) >= self.frame_bytes:
            frame = bytes(self._pending[:self.frame_bytes])
            del self._pending[:self.frame_bytes]
            self._total_frames += 1
            h = zlib.crc32(frame)
            if self._is_duplicate(h):
                self._dup_dropped += 1
                self._hashes.append(h)
                continue
            self._hashes.append(h)
            self._queue.append(frame)
        return self._drain()

    def _queue_ms(self) -> float:
        return (len(self._queue) * self.frame_bytes) / (self.sample_rate * 2) * 1000.0

    def _is_duplicate(self, h: int) -> bool:
        if not self._hashes:
            return False
        if h == self._hashes[-1]:
            return True
        if len(self._hashes) >= 3 and h == self._hashes[-2] and self._hashes[-1] == self._hashes[-3]:
            return True
        if len(self._hashes) >= 5 and h == self._hashes[-3]:
            if self._hashes[-1] == self._hashes[-4] and self._hashes[-2] == self._hashes[-5]:
                return True
        return False

    def _drain(self) -> List[bytes]:
        out: List[bytes] = []
        while self._queue_ms() > self.max_delay_ms:
            self._queue.popleft()
            self._old_dropped += 1
        while self._queue_ms() > self.target_delay_ms:
            out.append(self._queue.popleft())
        return out

    def stats(self) -> tuple[int, int, int]:
        return self._total_frames, self._dup_dropped, self._old_dropped
