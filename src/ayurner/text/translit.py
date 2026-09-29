"""Devanagari <-> IAST transliteration (thin wrapper over indic-transliteration)."""

from __future__ import annotations

import unicodedata
from functools import lru_cache

from indic_transliteration import sanscript

from .normalize import normalize_str
from .tokenize import is_devanagari

# ISO-15919 / Monier-Williams spellings that sanscript's IAST scheme does not expect.
_IAST_FIXES = {
    "ṁ": "ṃ",
    "Ṁ": "Ṃ",
    "m̐": "ṃ",
    "r̥": "ṛ",
}


def _fix_iast(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    for a, b in _IAST_FIXES.items():
        text = text.replace(a, b)
    return text


@lru_cache(maxsize=200_000)
def deva_to_iast(text: str) -> str:
    return sanscript.transliterate(normalize_str(text), sanscript.DEVANAGARI, sanscript.IAST)


@lru_cache(maxsize=200_000)
def iast_to_deva(text: str) -> str:
    return sanscript.transliterate(_fix_iast(text).lower(), sanscript.IAST, sanscript.DEVANAGARI)


def to_latin(text: str) -> str:
    """Return a lowercase Latin (IAST-ish) rendering of any supported input."""
    if is_devanagari(text):
        return deva_to_iast(text).lower()
    return _fix_iast(text).lower()


def transliterate(text: str, to: str = "iast") -> str:
    """Transliterate text between Devanagari and IAST.

    ``to`` is ``"iast"`` or ``"devanagari"``. Input already in the target
    script is returned unchanged (after normalization).
    """
    to = to.lower()
    if to == "iast":
        return deva_to_iast(text) if is_devanagari(text) else _fix_iast(text)
    if to in ("devanagari", "deva"):
        return normalize_str(text) if is_devanagari(text) else iast_to_deva(text)
    raise ValueError(f"unsupported target script: {to!r} (use 'iast' or 'devanagari')")
