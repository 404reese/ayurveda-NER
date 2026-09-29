"""Export documents as JSONL, CoNLL/BIO or Label Studio pre-annotations.

CoNLL and Label Studio output make the dictionary tagger a source of *silver*
training data: correct it in an annotation tool, then train an ML recognizer.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import IO

from .text.normalize import normalize
from .text.tokenize import tokenize
from .types import Document


def to_jsonl(docs: Iterable[Document], fh: IO[str]) -> int:
    n = 0
    for d in docs:
        fh.write(d.to_json())
        fh.write("\n")
        n += 1
    return n


def to_bio(doc: Document) -> list[tuple[str, str]]:
    """Token-level BIO tags: [(token_text, "B-DRAVYA" | "I-DRAVYA" | "O"), ...]."""
    norm = normalize(doc.text)
    out = []
    for tok in tokenize(norm.text):
        s, e = norm.to_original(tok.start, tok.end)
        tag = "O"
        for ent in doc.entities:
            if ent.start <= s and e <= ent.end:
                tag = ("B-" if s == ent.start else "I-") + ent.label
                break
        out.append((doc.text[s:e], tag))
    return out


def to_conll(docs: Iterable[Document], fh: IO[str]) -> int:
    n = 0
    for d in docs:
        for token, tag in to_bio(d):
            fh.write(f"{token}\t{tag}\n")
        fh.write("\n")
        n += 1
    return n


def to_label_studio(docs: Iterable[Document], model_version: str = "ayurner") -> list[dict]:
    """Label Studio tasks with predictions (import as JSON, labels config: <Labels name="label" toName="text">)."""
    tasks = []
    for d in docs:
        result = [
            {
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                "value": {"start": e.start, "end": e.end, "text": e.text, "labels": [e.label]},
                "score": e.confidence,
                "meta": {"entity_id": e.entity_id, "canonical": e.canonical},
            }
            for e in d.entities
        ]
        tasks.append(
            {
                "data": {"text": d.text},
                "predictions": [{"model_version": model_version, "result": result}],
            }
        )
    return tasks


def write(docs: Iterable[Document], path: str | Path, fmt: str | None = None) -> int:
    """Write documents to ``path``. Format from ``fmt`` or the extension (.jsonl, .conll, .json)."""
    path = Path(path)
    fmt = (fmt or path.suffix.lstrip(".")).lower()
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        if fmt == "jsonl":
            return to_jsonl(docs, fh)
        if fmt in ("conll", "bio", "tsv"):
            return to_conll(docs, fh)
        if fmt in ("json", "labelstudio", "label_studio"):
            tasks = to_label_studio(docs)
            json.dump(tasks, fh, ensure_ascii=False, indent=1)
            return len(tasks)
    raise ValueError(f"unknown export format: {fmt!r}")
