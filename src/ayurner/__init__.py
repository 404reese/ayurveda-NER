"""ayurner — named-entity recognition for Ayurvedic terms in Sanskrit and Hindi.

Quick start::

    import ayurner
    ner = ayurner.load()
    doc = ner("अश्वगंधा चूर्ण वात रोगों में उपयोगी है")
    for e in doc.entities:
        print(e.text, e.label, e.canonical)
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

from ._version import __version__
from .labels import all_labels, register_label
from .lexicon.index import build_cached
from .lexicon.loader import bundled_paths, load_bundled, load_path, merge
from .lexicon.schema import LexiconEntry
from .match.rules import RuleEngine
from .pipeline import Pipeline
from .text.normalize import normalize_str as normalize
from .text.translit import transliterate
from .types import Component, Document, Entity

__all__ = [
    "Component",
    "Document",
    "Entity",
    "LexiconEntry",
    "Pipeline",
    "__version__",
    "all_labels",
    "load",
    "normalize",
    "register_label",
    "transliterate",
]


def _env_lexicons() -> tuple[str, ...]:
    raw = os.environ.get("AYURNER_LEXICONS", "")
    return tuple(p for p in raw.split(os.pathsep) if p.strip())


@lru_cache(maxsize=8)
def _load_cached(
    lexicons: tuple[str, ...],
    include_core: bool,
    match_english: bool,
    fuzzy: bool,
    labels: frozenset[str] | None,
    rules: str | None,
    use_cache: bool,
    min_confidence: float,
) -> Pipeline:
    core = bundled_paths() if include_core else []
    files = [*core, *(Path(p) for p in lexicons)]
    for f in files:
        if not f.exists():
            raise FileNotFoundError(f"lexicon file not found: {f}")
    blobs = [f.name.encode() + b"\0" + f.read_bytes() for f in files]

    def entries():
        out = load_bundled() if include_core else []
        for p in lexicons:
            out.extend(load_path(p))
        return merge(out)

    index = build_cached(entries, blobs, match_english=match_english, use_cache=use_cache)
    fuzzy_rec = None
    if fuzzy:
        from .match.fuzzy import FuzzyRecognizer

        fuzzy_rec = FuzzyRecognizer()
    return Pipeline(
        index=index,
        rules=RuleEngine.from_file(rules),
        fuzzy=fuzzy_rec,
        labels=labels,
        min_confidence=min_confidence,
        config={
            "lexicons": list(lexicons),
            "include_core": include_core,
            "match_english": match_english,
            "fuzzy": fuzzy,
            "labels": sorted(labels) if labels else None,
            "rules": rules,
            "use_cache": use_cache,
            "min_confidence": min_confidence,
        },
    )


def load(
    lexicons: Iterable[str | Path] | None = None,
    *,
    include_core: bool = True,
    match_english: bool = True,
    fuzzy: bool = False,
    labels: Iterable[str] | None = None,
    rules: str | Path | None = None,
    use_cache: bool = True,
    min_confidence: float = 0.0,
) -> Pipeline:
    """Create (or reuse) an NER pipeline.

    Args:
        lexicons: extra lexicon files (.csv, .jsonl, or {category: [terms]} .json).
            Defaults to the paths in ``AYURNER_LEXICONS`` (os.pathsep-separated).
        include_core: include the bundled core lexicon.
        match_english: also match English equivalents ("Fever", "Piles").
        fuzzy: enable edit-distance fallback for misspellings (needs rapidfuzz).
        labels: only return entities with these labels.
        rules: path to a custom rules.toml (defaults to the bundled one).
        use_cache: reuse / write the compiled index in the user cache directory.
        min_confidence: drop entities below this confidence.

    Pipelines are memoized per configuration, so repeated calls are cheap.
    """
    paths = (
        tuple(str(Path(p).resolve()) for p in lexicons) if lexicons is not None else _env_lexicons()
    )
    return _load_cached(
        paths,
        include_core,
        match_english,
        fuzzy,
        frozenset(x.upper() for x in labels) if labels else None,
        str(Path(rules).resolve()) if rules else None,
        use_cache,
        float(min_confidence),
    )
