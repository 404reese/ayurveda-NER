"""Optional fuzzy fallback for misspelled single words (requires ``rapidfuzz``)."""

from __future__ import annotations

from collections import defaultdict

from .base import Candidate, MatchContext, Sense
from .gazetteer import token_alts


class FuzzyRecognizer:
    """Edit-distance matching of unmatched long tokens against single-word lexicon keys.

    Candidates are bucketed by first letter and length so each lookup scans a
    small slice of the vocabulary. Off by default; enable with ``load(fuzzy=True)``.
    """

    def __init__(self, min_len: int = 6, threshold: float = 88.0):
        try:
            from rapidfuzz import fuzz, process
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "fuzzy matching needs rapidfuzz: pip install 'ayurner[fuzzy]'"
            ) from exc
        self._process, self._scorer = process, fuzz.ratio
        self.min_len = min_len
        self.threshold = threshold
        self._buckets: dict[int, dict[tuple[str, int], list[str]]] = {}

    def _bucketed(self, index):
        b = self._buckets.get(id(index))
        if b is None:
            b = defaultdict(list)
            for key, hits in index.phrases.items():
                if (
                    len(key) == 1
                    and len(key[0]) >= self.min_len - 1
                    and any(not h.weak for h in hits)
                ):
                    b[(key[0][0], len(key[0]))].append(key[0])
            self._buckets = {id(index): b}
        return b

    def recognize(self, ctx: MatchContext, covered: set[int] | None = None) -> list[Candidate]:
        covered = covered or set()
        buckets = self._bucketed(ctx.index)
        out = []
        for i, tok in enumerate(ctx.tokens):
            if i in covered:
                continue
            key = token_alts(tok.text).strict[0][0]
            if len(key) < self.min_len:
                continue
            pool = []
            for d in (-1, 0, 1):
                pool.extend(buckets.get((key[0], len(key) + d), ()))
            if not pool:
                continue
            best = self._process.extractOne(
                key, pool, scorer=self._scorer, score_cutoff=self.threshold
            )
            if not best:
                continue
            hits = [h for h in ctx.index.phrases[(best[0],)] if not h.weak and not h.english]
            if not hits:
                continue
            senses = [Sense(ctx.index.entries[h.entry]) for h in hits]
            out.append(
                Candidate(i, i + 1, senses, "fuzzy", round(best[1] / 100 * 0.8, 3), source="fuzzy")
            )
        return out
