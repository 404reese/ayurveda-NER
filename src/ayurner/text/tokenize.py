"""Script-aware tokenizer for Devanagari and (IAST / casual) Latin text."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Devanagari letters and signs (excluding danda U+0964/5 and digits U+0966-096F),
# Devanagari Extended and Vedic Extensions.
_DEVA = r"ऀ-ॣ॰-ॿ꣠-ꣿ᳐-᳿"
# Latin letters incl. IAST diacritics (Latin-1 letters, Extended-A/B,
# Extended Additional) and combining marks, plus apostrophe for avagraha.
_LATIN = r"A-Za-zÀ-ÖØ-öø-ɏḀ-ỿ̀-ͯ'"

_TOKEN_RE = re.compile(rf"[{_DEVA}]+|[{_LATIN}]+")
_DEVA_RE = re.compile(rf"[{_DEVA}]")
# Characters that end a segment (sentence / verse half).
_SEGMENT_BREAK_RE = re.compile(r"[।॥.!?;|\n]")
# Gaps that still allow two tokens to form one multi-word term.
_JOINABLE_GAP_RE = re.compile(r"^[\s\-]*$")


@dataclass(frozen=True, slots=True)
class Token:
    text: str
    start: int  # offsets into the *normalized* text
    end: int
    index: int
    segment: int
    joinable: bool  # True if only whitespace / hyphens separate it from the previous token
    is_devanagari: bool


def is_devanagari(text: str) -> bool:
    return bool(_DEVA_RE.search(text))


def tokenize(text: str) -> list[Token]:
    """Split normalized text into word tokens with offsets and segment ids."""
    tokens: list[Token] = []
    segment = 0
    prev_end = 0
    for m in _TOKEN_RE.finditer(text):
        word = m.group().strip("'")
        if not word:
            continue
        start = m.start() + (len(m.group()) - len(m.group().lstrip("'")))
        end = start + len(word)
        gap = text[prev_end:start]
        if tokens and _SEGMENT_BREAK_RE.search(gap):
            segment += 1
        joinable = bool(tokens) and bool(_JOINABLE_GAP_RE.match(gap))
        tokens.append(
            Token(
                text=word,
                start=start,
                end=end,
                index=len(tokens),
                segment=segment,
                joinable=joinable,
                is_devanagari=is_devanagari(word),
            )
        )
        prev_end = end
    return tokens


def split_words(text: str) -> list[str]:
    """Tokenize a lexicon surface form into plain word strings."""
    return [t.text for t in tokenize(text)]
