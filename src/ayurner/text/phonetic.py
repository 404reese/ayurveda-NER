"""Phonetic match keys: one key space for Devanagari, IAST and casual romanization.

Every lexicon surface form and every input token is folded into a *key*.
Two spellings that a reader would consider "the same word" should produce the
same key, e.g. अश्वगन्धा / अश्वगंधा / aśvagandhā / ashwagandha / ashvagandh.

Two tiers:

* **strict** keys remove script and orthography noise (diacritics, vowel
  length, nasal spelling, doubled consonants, visarga, the Hindi final schwa,
  casual spellings such as ``w``/``v``, ``ch``/``c``, ``ee``/``i``).
* **loose** keys additionally drop aspiration and the ``sh``/``s`` contrast.
  They are only used for casual ASCII romanizations, where writers are
  inconsistent about these (``Vatha``/``Vata``, ``Swasa``/``Shvasa``).

Bump ``KEY_VERSION`` whenever folding changes so compiled indexes are rebuilt.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

from .tokenize import is_devanagari
from .translit import to_latin

KEY_VERSION = 3

VOWELS = frozenset("aeiou")

_IAST_MAP = (
    ("ṝ", "ri"),
    ("ṛ", "ri"),
    ("ḹ", "li"),
    ("ḷ", "li"),
    ("ś", "sh"),
    ("ṣ", "sh"),
    ("ṭ", "t"),
    ("ḍ", "d"),
    ("ṇ", "n"),
    ("ñ", "n"),
    ("ṅ", "n"),
    ("ḥ", ""),
    ("ṃ", "M"),
    ("~", "M"),
    ("r̤", "r"),
    ("ḻ", "l"),
)

_CASUAL_MAP = (
    ("x", "ksh"),
    ("q", "k"),
    ("w", "v"),
    ("z", "j"),
    ("ph", "f"),
    ("chh", "c"),
    ("ch", "c"),
    ("ee", "i"),
    ("oo", "u"),
    ("ou", "au"),
)

_LOOSE_MAP = (
    ("kh", "k"),
    ("gh", "g"),
    ("jh", "j"),
    ("th", "t"),
    ("dh", "d"),
    ("bh", "b"),
    ("f", "p"),
    ("sh", "s"),
)

_NON_KEY_RE = re.compile(r"[^a-zM]")
_NASAL_BEFORE_CONS_RE = re.compile(r"[Mmn](?=[b-df-hj-np-tv-z])")
_DOUBLE_RE = re.compile(r"(.)\1+")
_ASCII_WORD_RE = re.compile(r"^[A-Za-z'\-]+$")


def _strip_marks(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c))


def _fold_latin(latin: str, *, loose: bool, drop_final_schwa: bool) -> str:
    s = latin.lower()
    for a, b in _IAST_MAP:
        s = s.replace(a, b)
    s = _strip_marks(s)
    s = _NON_KEY_RE.sub("", s)
    for a, b in _CASUAL_MAP:
        s = s.replace(a, b)
    s = _NASAL_BEFORE_CONS_RE.sub("n", s)
    s = s.replace("M", "m")
    if loose:
        for a, b in _LOOSE_MAP:
            s = s.replace(a, b)
    s = _DOUBLE_RE.sub(r"\1", s)
    if drop_final_schwa:
        s = drop_schwa(s)
    return s


def drop_schwa(key: str) -> str:
    """Drop a word-final inherent ``a`` (vāta → vat, as Hindi speakers say it)."""
    if len(key) >= 3 and key[-1] == "a" and key[-2] not in VOWELS:
        return key[:-1]
    return key


def is_casual_latin(word: str) -> bool:
    """True for plain-ASCII romanizations (no IAST diacritics, not Devanagari)."""
    return bool(_ASCII_WORD_RE.match(word))


@lru_cache(maxsize=500_000)
def strict_key(word: str, drop_final_schwa: bool = True) -> str:
    return _fold_latin(to_latin(word), loose=False, drop_final_schwa=drop_final_schwa)


@lru_cache(maxsize=500_000)
def loose_key(word: str, drop_final_schwa: bool = True) -> str:
    return _fold_latin(to_latin(word), loose=True, drop_final_schwa=drop_final_schwa)


def loosen(key: str) -> str:
    """Convert a strict key into a loose one."""
    return _fold_latin(key, loose=True, drop_final_schwa=True)


# --- Hindi medial schwa deletion (lexicon-side variants) -------------------

_IAST_VOWEL_UNITS = frozenset(
    ("ai", "au", "a", "ā", "i", "ī", "u", "ū", "ṛ", "ṝ", "ḷ", "ḹ", "e", "o")
)
_IAST_ASPIRATES = frozenset(("kh", "gh", "ch", "jh", "ṭh", "ḍh", "th", "dh", "ph", "bh"))


def _units(word: str) -> list[str]:
    units: list[str] = []
    i = 0
    while i < len(word):
        two = word[i : i + 2]
        if two in ("ai", "au") or two in _IAST_ASPIRATES:
            units.append(two)
            i += 2
        else:
            units.append(word[i])
            i += 1
    return units


def _is_vowel(u: str) -> bool:
    return u in _IAST_VOWEL_UNITS


def schwa_delete_iast(iast: str) -> str:
    """Apply Hindi schwa deletion (Ohala's rule, right to left) to an IAST string.

    A short ``a`` is dropped in the context V C+ _ C V, e.g. dālacīnī → dālcīnī,
    ajavāina → ajvāina, nāgaramothā → nāgarmothā, yogarāja → yograja.
    Long vowels are never deleted, which is why this works on IAST rather than
    on folded keys.
    """
    words = []
    for word in unicodedata.normalize("NFC", iast.lower()).split():
        u = _units(word)
        i = len(u) - 3
        while i >= 1:
            if (
                u[i] == "a"
                and not _is_vowel(u[i - 1])
                and not _is_vowel(u[i + 1])
                and _is_vowel(u[i + 2])
            ):
                j = i - 1
                while j >= 0 and not _is_vowel(u[j]):
                    j -= 1
                if j >= 0:
                    del u[i]
            i -= 1
        words.append("".join(u))
    return " ".join(words)


def lexicon_token_keys(word: str) -> set[str]:
    """All strict keys a lexicon surface token should be indexed under."""
    latin = to_latin(word)
    keys = {_fold_latin(latin, loose=False, drop_final_schwa=True)}
    if is_devanagari(word) or not is_casual_latin(word):
        deleted = schwa_delete_iast(latin)
        if deleted != latin:
            keys.add(_fold_latin(deleted, loose=False, drop_final_schwa=True))
    full = _fold_latin(latin, loose=False, drop_final_schwa=False)
    # Hindi often drops a Sanskrit final -u (guggulu → guggul).
    if len(full) >= 5 and full[-1] == "u" and full[-2] not in VOWELS:
        keys.add(full[:-1])
    keys.discard("")
    return keys
