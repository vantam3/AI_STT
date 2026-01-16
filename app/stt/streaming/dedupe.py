class TextDedupe:
    def __init__(self):
        self._last_emitted_norm = ""
        self._last_bigram = None
        self._bigram_hits = 0

    @staticmethod
    def _norm(text: str) -> str:
        return " ".join((text or "").strip().lower().split())

    def is_duplicate(self, text: str) -> bool:
        new_norm = self._norm(text)
        prev_norm = self._last_emitted_norm
        if not new_norm:
            return True
        if new_norm == prev_norm:
            return True
        lcp = 0
        n = min(len(new_norm), len(prev_norm))
        while lcp < n and new_norm[lcp] == prev_norm[lcp]:
            lcp += 1
        if n > 0 and (lcp / max(len(new_norm), len(prev_norm))) >= 0.9:
            return True
        return False

    def is_repetitive(self, text: str) -> bool:
        words = self._norm(text).split()
        if len(words) < 4:
            return False
        bigram = (words[-2], words[-1])
        if self._last_bigram == bigram:
            self._bigram_hits += 1
        else:
            self._bigram_hits = 1
            self._last_bigram = bigram
        if self._bigram_hits >= 4:
            return True
        run = 1
        for i in range(1, len(words)):
            if words[i] == words[i - 1]:
                run += 1
                if run >= 3:
                    return True
            else:
                run = 1
        unique = len(set(words))
        if unique <= 2 and len(words) >= 8:
            return True
        return False

    def update(self, text: str) -> None:
        self._last_emitted_norm = self._norm(text)
