"""Overlap resolution: keep the best non-overlapping set of candidates."""

from __future__ import annotations

from .base import Candidate

_TYPE_RANK = {
    "exact": 0,
    "phonetic": 1,
    "compose": 2,
    "stem": 3,
    "compound": 4,
    "loose": 5,
    "fuzzy": 6,
}


def resolve(cands: list[Candidate]) -> list[Candidate]:
    """Greedy selection: longer spans first, then stronger (non-weak), better match type, confidence."""
    ranked = sorted(
        cands,
        key=lambda c: (
            -c.length,
            c.weak or c.compose_only,
            _TYPE_RANK.get(c.match_type, 9),
            -c.confidence,
            c.tok_start,
        ),
    )
    taken: set[int] = set()
    out: list[Candidate] = []
    for c in ranked:
        span = range(c.tok_start, c.tok_end)
        if any(i in taken for i in span):
            continue
        taken.update(span)
        out.append(c)
    return sorted(out, key=lambda c: c.tok_start)
