"""Lexicon entry schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# requires_context values
CONTEXT_NONE = ""
CONTEXT_YES = "yes"  # canonical + Devanagari forms need a nearby cue
CONTEXT_COMPOSE = "compose"  # never emitted alone; only as part of a compound/composition

# Labels whose single-word English equivalents are specific enough to match without context.
STRONG_ENGLISH_LABELS = frozenset({"ROGA", "FORMULATION", "PROCEDURE"})

CSV_COLUMNS = (
    "id",
    "label",
    "subtype",
    "canonical_iast",
    "devanagari",
    "synonyms",
    "hindi",
    "english",
    "definition",
    "botanical",
    "xrefs",
    "requires_context",
    "notes",
    "source",
)


@dataclass(frozen=True, slots=True)
class SurfaceForm:
    text: str
    weak: bool = False  # needs context to be accepted
    english: bool = False


@dataclass(frozen=True, slots=True)
class LexiconEntry:
    id: str
    label: str
    canonical: str  # IAST
    devanagari: str
    subtype: str = ""
    synonyms: tuple[str, ...] = ()
    hindi: tuple[str, ...] = ()
    english: tuple[str, ...] = ()
    definition: str = ""
    botanical: str = ""
    xrefs: tuple[str, ...] = ()
    requires_context: str = CONTEXT_NONE
    notes: str = ""
    source: str = ""
    extra: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)

    @property
    def compose_only(self) -> bool:
        return self.requires_context == CONTEXT_COMPOSE

    def surface_forms(self, match_english: bool = True) -> list[SurfaceForm]:
        """Every string this entry can be recognized by.

        A leading ``~`` on a synonym / Hindi / English form marks it as weak
        (needs context). ``requires_context=yes`` makes the canonical and
        Devanagari forms weak.
        """
        head_weak = self.requires_context in (CONTEXT_YES, CONTEXT_COMPOSE)
        all_weak = self.requires_context == CONTEXT_COMPOSE
        forms: list[SurfaceForm] = []
        seen: set[str] = set()

        def add(text: str, weak: bool, english: bool = False) -> None:
            text = text.strip()
            if text.startswith("~"):
                text, weak = text[1:].strip(), True
            if not text or text.lower() in seen:
                return
            seen.add(text.lower())
            forms.append(SurfaceForm(text, weak or all_weak, english))

        add(self.canonical, head_weak)
        add(self.devanagari, head_weak)
        for s in (*self.synonyms, *self.hindi):
            add(s, False)
        if match_english:
            for s in self.english:
                # Single English words ("Hard", "Honey", "Oil") are everyday vocabulary;
                # only specific labels keep them strong.
                weak = " " not in s.strip() and not self._specific_english()
                add(s, weak, english=True)
        return forms

    def _specific_english(self) -> bool:
        if self.label in STRONG_ENGLISH_LABELS:
            return True
        return self.label == "DRAVYA" and self.subtype == "herb"

    def metadata(self) -> dict[str, Any]:
        md: dict[str, Any] = {}
        if self.english:
            md["english"] = [e.lstrip("~") for e in self.english]
        if self.hindi:
            md["hindi"] = [h.lstrip("~") for h in self.hindi]
        if self.botanical:
            md["botanical"] = self.botanical
        if self.xrefs:
            md["xrefs"] = list(self.xrefs)
        if self.definition:
            md["definition"] = self.definition
        if self.source:
            md["source"] = self.source
        return md

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "subtype": self.subtype,
            "canonical": self.canonical,
            "devanagari": self.devanagari,
            "synonyms": [s.lstrip("~") for s in self.synonyms],
            **self.metadata(),
        }
