"""Key-level stemming for Sanskrit case endings and Hindi plural/oblique forms.

Works on folded keys, so one suffix table covers Devanagari, IAST and casual
romanization (रोगेषु / rogeṣu / rogeshu all strip ``eshu``).
"""

from __future__ import annotations

import sys
from functools import lru_cache
from importlib import resources

from .phonetic import drop_schwa

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib


@lru_cache(maxsize=1)
def _suffix_table() -> tuple[tuple[str, ...], int]:
    raw = resources.files("ayurner.lexicon.data").joinpath("suffixes.toml").read_bytes()
    data = tomllib.loads(raw.decode("utf-8"))
    suffixes = set(data.get("sanskrit", [])) | set(data.get("hindi", []))
    ordered = tuple(sorted(suffixes, key=len, reverse=True))
    return ordered, int(data.get("min_stem_length", 3))


def suffixes() -> tuple[str, ...]:
    return _suffix_table()[0]


@lru_cache(maxsize=200_000)
def stem_candidates(full_key: str) -> tuple[str, ...]:
    """Candidate stems (schwa-dropped) for a strict key *without* final-schwa drop.

    The unstripped key is not included; callers try it first.
    """
    table, min_len = _suffix_table()
    out: list[str] = []
    seen: set[str] = set()
    for suf in table:
        if full_key.endswith(suf) and len(full_key) - len(suf) >= min_len:
            base = full_key[: -len(suf)]
            for cand in (drop_schwa(base), base):
                if cand not in seen and len(cand) >= min_len:
                    seen.add(cand)
                    out.append(cand)
    return tuple(out)


def strip_suffix(full_key: str) -> tuple[str, str] | None:
    """Return (stem, suffix) if ``full_key`` is exactly a known suffix-bearing form."""
    table, _ = _suffix_table()
    for suf in table:
        if full_key.endswith(suf):
            return full_key[: -len(suf)], suf
    return None


def is_suffix(fragment: str) -> bool:
    return fragment in set(_suffix_table()[0])
