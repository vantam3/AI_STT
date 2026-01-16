from dataclasses import dataclass


def longest_common_prefix(a: str, b: str) -> str:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return a[:i]


@dataclass
class Stabilized:
    stable: str
    partial: str


class SimpleStabilizer:
    def __init__(self, min_chars: int = 18):
        self.min_chars = min_chars
        self._stable = ""
        self._prev = ""

    def update(self, new_text: str) -> Stabilized:
        new_text = (new_text or "").strip()
        if not new_text:
            self._prev = ""
            return Stabilized(stable=self._stable, partial="")

        if not self._prev:
            self._prev = new_text
            return Stabilized(stable=self._stable, partial=new_text)

        lcp = longest_common_prefix(self._prev, new_text)
        if len(lcp) >= self.min_chars:
            if len(self._stable) < len(lcp):
                self._stable = lcp.strip()
        self._prev = new_text

        partial = new_text[len(self._stable):].lstrip() if self._stable else new_text
        return Stabilized(stable=self._stable, partial=partial)

    def reset(self) -> None:
        self._stable = ""
        self._prev = ""
