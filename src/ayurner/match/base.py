"""Internal candidate representation shared by recognizers, rules and the resolver."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..lexicon.index import LexiconIndex
from ..lexicon.schema import LexiconEntry
from ..text.normalize import NormalizedText
from ..text.tokenize import Token

BASE_CONFIDENCE = {
    "exact": 1.0,
    "phonetic": 0.95,
    "stem": 0.9,
    "compose": 0.9,
    "loose": 0.85,
    "compound": 0.85,
}


@dataclass(frozen=True, slots=True)
class Sense:
    entry: LexiconEntry
    weak: bool = False


@dataclass
class Candidate:
    tok_start: int
    tok_end: int  # exclusive
    senses: list[Sense]
    match_type: str
    confidence: float
    components: list[Candidate] = field(default_factory=list)
    # Set when the candidate is a synthesized composition with no lexicon entry.
    label: str | None = None
    subtype: str = ""
    ambiguous: bool = False
    source: str = "gazetteer"

    @property
    def length(self) -> int:
        return self.tok_end - self.tok_start

    @property
    def weak(self) -> bool:
        return bool(self.senses) and all(s.weak for s in self.senses)

    @property
    def compose_only(self) -> bool:
        return bool(self.senses) and all(s.entry.compose_only for s in self.senses)

    @property
    def primary(self) -> LexiconEntry | None:
        return self.senses[0].entry if self.senses else None

    @property
    def final_label(self) -> str:
        if self.label:
            return self.label
        return self.primary.label if self.primary else ""

    @property
    def final_subtype(self) -> str:
        if self.label:
            return self.subtype
        return self.primary.subtype if self.primary else ""

    def labels(self) -> set[str]:
        return {s.entry.label for s in self.senses}


@dataclass
class MatchContext:
    """Everything a recognizer needs for one document."""

    normalized: NormalizedText
    tokens: list[Token]
    index: LexiconIndex


class Recognizer(Protocol):
    """Anything that proposes candidates. A future ML model implements this too."""

    def recognize(self, ctx: MatchContext) -> list[Candidate]: ...
