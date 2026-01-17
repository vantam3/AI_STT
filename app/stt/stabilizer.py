from dataclasses import dataclass


def longest_common_prefix(a: str, b: str) -> str:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return a[:i]


def _trim_to_word_boundary(s: str) -> str:
    s = s.strip()
    if not s:
        return ""
    # cắt về ranh giới từ cuối cùng để tránh commit nửa chữ
    cut = s.rfind(" ")
    if cut <= 0:
        return s
    return s[:cut].strip()


@dataclass
class Stabilized:
    stable: str
    partial: str


class SimpleStabilizer:
    """
    Stabilizer kiểu LCP nhưng:
    - chỉ commit khi LCP đủ dài
    - commit theo word-boundary (tránh "Hoàn to")
    - nếu stable không còn là prefix của new_text => reset stable
    """

    def __init__(self, min_chars: int = 18):
        self.min_chars = int(min_chars)
        self._stable = ""
        self._prev = ""

    def update(self, new_text: str) -> Stabilized:
        new_text = (new_text or "").strip()

        if not new_text:
            self._prev = ""
            return Stabilized(stable=self._stable, partial="")

        # lần đầu
        if not self._prev:
            self._prev = new_text
            return Stabilized(stable=self._stable, partial=new_text)

        # nếu stable hiện tại không còn match prefix nữa => reset (tránh lệch)
        if self._stable and not new_text.startswith(self._stable):
            self._stable = ""

        lcp = longest_common_prefix(self._prev, new_text)

        if len(lcp) >= self.min_chars:
            candidate = _trim_to_word_boundary(lcp)
            if candidate and len(candidate) > len(self._stable):
                self._stable = candidate

        self._prev = new_text

        if self._stable and new_text.startswith(self._stable):
            partial = new_text[len(self._stable):].lstrip()
        else:
            partial = new_text

        return Stabilized(stable=self._stable, partial=partial)

    def reset(self) -> None:
        self._stable = ""
        self._prev = ""
