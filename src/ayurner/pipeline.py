"""The NER pipeline: normalize → tokenize → recognize → resolve → rules → entities."""

from __future__ import annotations

import itertools
from collections import deque
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from .lexicon.index import LexiconIndex
from .lexicon.schema import LexiconEntry
from .match.base import Candidate, MatchContext, Recognizer
from .match.gazetteer import GazetteerRecognizer, token_alts
from .match.resolve import resolve
from .match.rules import RuleEngine
from .text.normalize import normalize
from .text.phonetic import strict_key
from .text.tokenize import split_words, tokenize
from .types import Component, Document, Entity

AMBIGUITY_PENALTY = 0.85


@dataclass
class Pipeline:
    """Callable NER pipeline. Build one with :func:`ayurner.load`."""

    index: LexiconIndex
    recognizers: list[Recognizer] = field(default_factory=lambda: [GazetteerRecognizer()])
    rules: RuleEngine = field(default_factory=RuleEngine.from_file)
    fuzzy: Any = None
    labels: frozenset[str] | None = None
    min_confidence: float = 0.0
    config: dict[str, Any] | None = (
        None  # set by load(); lets worker processes rebuild the pipeline
    )

    # ------------------------------------------------------------ building
    @classmethod
    def from_entries(
        cls, entries: Iterable[LexiconEntry], *, match_english: bool = True, **kwargs
    ) -> Pipeline:
        """Build an uncached pipeline straight from entries (handy for tests and quick experiments)."""
        return cls(index=LexiconIndex.build(list(entries), match_english=match_english), **kwargs)

    def add_recognizer(self, recognizer: Recognizer) -> None:
        """Append a recognizer (e.g. a future ML model). Its candidates go through the same rules."""
        self.recognizers.append(recognizer)

    @property
    def lexicon_version(self) -> str:
        return self.index.version or "uncached"

    # ------------------------------------------------------------- running
    def __call__(self, text: str) -> Document:
        norm = normalize(text)
        tokens = tokenize(norm.text)
        ctx = MatchContext(norm, tokens, self.index)
        cands: list[Candidate] = []
        for r in self.recognizers:
            cands.extend(r.recognize(ctx))
        if self.fuzzy is not None:
            covered = {i for c in cands for i in range(c.tok_start, c.tok_end)}
            cands.extend(self.fuzzy.recognize(ctx, covered))
        cands = resolve(cands)
        self.rules.disambiguate(cands, tokens)
        cands = self.rules.compose(cands, tokens, self.index, norm.text)
        cands = self.rules.filter_context(cands, tokens)

        entities = []
        for c in cands:
            ent = self._to_entity(c, norm, tokens)
            if self.labels and ent.label not in self.labels:
                continue
            if ent.confidence < self.min_confidence:
                continue
            entities.append(ent)
        return Document(text=text, entities=tuple(entities), lexicon_version=self.lexicon_version)

    def extract(self, text: str) -> list[Entity]:
        return list(self(text).entities)

    def pipe(
        self, texts: Iterable[str], batch_size: int = 64, n_process: int = 1
    ) -> Iterator[Document]:
        """Process many texts lazily, in order. ``n_process > 1`` uses worker processes.

        With multiple processes, call this under ``if __name__ == "__main__":``
        (required on Windows / macOS, which start workers by spawning).
        """
        if n_process <= 1:
            for t in texts:
                yield self(t)
            return
        if self.config is None:
            raise RuntimeError("multi-process pipe() needs a pipeline created with ayurner.load()")
        with ProcessPoolExecutor(
            n_process, initializer=_init_worker, initargs=(self.config,)
        ) as ex:
            pending: deque = deque()
            it = iter(texts)
            while True:
                chunk = list(itertools.islice(it, batch_size))
                if not chunk:
                    break
                pending.append(ex.submit(_process_chunk, chunk))
                if len(pending) >= n_process * 2:
                    yield from pending.popleft().result()
            while pending:
                yield from pending.popleft().result()

    # -------------------------------------------------------------- lookup
    def lookup(self, query: str) -> list[LexiconEntry]:
        """Find lexicon entries by any name, in any script (``"giloy"``, ``"गुडूची"``, ``"guḍūcī"``)."""
        words = split_words(normalize(query).text)
        if not words:
            return []
        found: dict[str, LexiconEntry] = {}

        def add(hits) -> None:
            for h in hits or ():
                e = self.index.entries[h.entry]
                found.setdefault(e.id, e)

        per_token = [[k for k, _ in token_alts(w).strict] for w in words]
        for combo in itertools.islice(itertools.product(*per_token), 64):
            add(self.index.phrases.get(tuple(combo)))
        if not found:
            add(self.index.joined.get(strict_key("".join(words))))
        if not found:
            loose = [[k for k, _ in token_alts(w).loose] for w in words]
            if all(loose):
                for combo in itertools.islice(itertools.product(*loose), 64):
                    add(self.index.loose_phrases.get(tuple(combo)))
        by_id = self.index.get(query.strip())
        if by_id:
            found.setdefault(by_id.id, by_id)
        return list(found.values())

    # ------------------------------------------------------------ internals
    @staticmethod
    def _component(c: Candidate) -> Component:
        if c.senses and not c.label:
            e = c.primary
            return Component(e.id, e.label, e.subtype, e.canonical)
        parts = [Pipeline._component(p) for p in c.components]
        return Component(
            "composite:" + "+".join(p.entity_id for p in parts),
            c.final_label,
            c.final_subtype,
            " ".join(p.canonical for p in parts),
        )

    def _to_entity(self, c: Candidate, norm, tokens) -> Entity:
        s_norm, e_norm = tokens[c.tok_start].start, tokens[c.tok_end - 1].end
        start, end = norm.to_original(s_norm, e_norm)
        components = tuple(self._component(p) for p in c.components)
        confidence = c.confidence * (AMBIGUITY_PENALTY if c.ambiguous else 1.0)
        if c.senses and not c.label:
            e = c.primary
            alternatives = tuple(
                dict.fromkeys(s.entry.id for s in c.senses[1:] if s.entry.id != e.id)
            )
            return Entity(
                text=norm.original[start:end],
                start=start,
                end=end,
                label=e.label,
                subtype=e.subtype,
                entity_id=e.id,
                canonical=e.canonical,
                devanagari=e.devanagari,
                confidence=round(confidence, 3),
                match_type=c.match_type,
                ambiguous=c.ambiguous,
                alternatives=alternatives,
                components=components,
                metadata=e.metadata(),
            )
        entries = [self.index.get(p.entity_id) for p in components]
        sep = "" if c.match_type == "compound" else " "
        return Entity(
            text=norm.original[start:end],
            start=start,
            end=end,
            label=c.final_label,
            subtype=c.final_subtype,
            entity_id="composite:" + "+".join(p.entity_id for p in components),
            canonical=sep.join(p.canonical for p in components),
            devanagari=sep.join(
                e.devanagari if e else p.canonical for e, p in zip(entries, components, strict=True)
            ),
            confidence=round(confidence, 3),
            match_type=c.match_type,
            ambiguous=c.ambiguous,
            components=components,
        )


# ------------------------------------------------------------ worker helpers
_WORKER: Pipeline | None = None


def _init_worker(config: dict[str, Any]) -> None:
    global _WORKER
    from . import load

    _WORKER = load(**config)


def _process_chunk(texts: list[str]) -> list[Document]:
    assert _WORKER is not None
    return [_WORKER(t) for t in texts]
