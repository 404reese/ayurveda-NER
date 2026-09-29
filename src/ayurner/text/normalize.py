"""Unicode normalization that remembers where every character came from.

Matching works on a normalized copy of the input, but entity spans must point
into the *original* string. ``normalize`` therefore returns, for every
normalized character, the ``[start, end)`` range of original characters it was
produced from.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

NUKTA = "़"
CHANDRABINDU = "ँ"
ANUSVARA = "ं"

# Characters removed outright: joiners, soft hyphen, BOM, Devanagari nukta.
_DROP = frozenset({"‌", "‍", "­", "﻿", NUKTA})

# One-to-one replacements applied after NFC.
_REPLACE = {
    CHANDRABINDU: ANUSVARA,
    "ॱ": "",  # Devanagari high spacing dot
    "‘": "'",
    "’": "'",
    "–": "-",
    "‐": "-",
    "‑": "-",
}


@dataclass(frozen=True, slots=True)
class NormalizedText:
    """Normalized text plus a per-character map back to the original."""

    text: str
    original: str
    starts: tuple[int, ...]
    ends: tuple[int, ...]

    def to_original(self, start: int, end: int) -> tuple[int, int]:
        """Map a normalized ``[start, end)`` span to the original string."""
        if end <= start:
            pos = self.starts[start] if start < len(self.starts) else len(self.original)
            return pos, pos
        return self.starts[start], self.ends[end - 1]


def _clusters(text: str):
    """Yield (start, end) runs of a base character followed by combining marks."""
    n = len(text)
    i = 0
    while i < n:
        j = i + 1
        while j < n and unicodedata.combining(text[j]):
            j += 1
        yield i, j
        i = j


def normalize(text: str) -> NormalizedText:
    """NFC-normalize and fold Devanagari spelling variants, tracking offsets.

    Folds applied: nukta removed (ड़ → ड, ज़ → ज), chandrabindu → anusvara,
    zero-width joiners / soft hyphens removed, typographic quotes and dashes
    simplified.
    """
    out: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    for s, e in _clusters(text):
        chunk = unicodedata.normalize("NFC", text[s:e])
        for ch in chunk:
            if ch in _DROP:
                continue
            ch = _REPLACE.get(ch, ch)
            if not ch:
                continue
            out.append(ch)
            starts.append(s)
            ends.append(e)
    return NormalizedText("".join(out), text, tuple(starts), tuple(ends))


def normalize_str(text: str) -> str:
    """Normalize without keeping the offset map."""
    return normalize(text).text
