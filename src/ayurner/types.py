"""Public result types."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Component:
    """One part of a compound or composed entity (e.g. ``aśvagandhā`` in ``aśvagandhā cūrṇa``)."""

    entity_id: str
    label: str
    subtype: str
    canonical: str


@dataclass(frozen=True, slots=True)
class Entity:
    """A recognized Ayurvedic term. ``start``/``end`` index into the original text."""

    text: str
    start: int
    end: int
    label: str
    entity_id: str
    canonical: str
    devanagari: str
    subtype: str = ""
    confidence: float = 1.0
    match_type: str = "exact"
    ambiguous: bool = False
    alternatives: tuple[str, ...] = ()
    components: tuple[Component, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Document:
    """The result of running the pipeline on one text."""

    text: str
    entities: tuple[Entity, ...]
    lexicon_version: str = ""

    def __iter__(self):
        return iter(self.entities)

    def __len__(self) -> int:
        return len(self.entities)

    def by_label(self, label: str) -> list[Entity]:
        return [e for e in self.entities if e.label == label]

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "lexicon_version": self.lexicon_version,
            "entities": [e.to_dict() for e in self.entities],
        }

    def to_json(self, **kwargs: Any) -> str:
        kwargs.setdefault("ensure_ascii", False)
        return json.dumps(self.to_dict(), **kwargs)
