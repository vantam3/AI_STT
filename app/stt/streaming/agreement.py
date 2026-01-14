class LocalAgreement:
    def __init__(self, min_hits: int):
        self.min_hits = min_hits
        self._last_stable = ""
        self._hits = 0

    def update(self, stable_text: str) -> int:
        if stable_text and stable_text == self._last_stable:
            self._hits += 1
        else:
            self._hits = 1 if stable_text else 0
            self._last_stable = stable_text
        return self._hits

    def hits(self) -> int:
        return self._hits

    def should_emit_final(self, stable_text: str) -> bool:
        return bool(stable_text) and self._hits >= self.min_hits
