"""Load lexicon entries from CSV (authoring format), JSONL, or quick term lists."""

from __future__ import annotations

import csv
import json
import logging
import re
from collections.abc import Iterable, Mapping
from importlib import resources
from pathlib import Path

from ..labels import is_known, register_label
from ..text.normalize import normalize_str
from ..text.tokenize import is_devanagari
from ..text.translit import deva_to_iast, iast_to_deva
from .schema import CONTEXT_COMPOSE, CONTEXT_NONE, CONTEXT_YES, LexiconEntry

log = logging.getLogger(__name__)

_SPLIT = "|"
_SLUG_RE = re.compile(r"[^a-z0-9]+")


class LexiconError(ValueError):
    pass


def _split(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(normalize_str(v.strip()) for v in value.split(_SPLIT) if v.strip())


def _context(value: str | None) -> str:
    v = (value or "").strip().lower()
    if v in ("", "no", "false", "0"):
        return CONTEXT_NONE
    if v in ("yes", "true", "1", "y"):
        return CONTEXT_YES
    if v == "compose":
        return CONTEXT_COMPOSE
    raise LexiconError(f"invalid requires_context value: {value!r}")


def slugify(text: str) -> str:
    from ..text.phonetic import strict_key

    return _SLUG_RE.sub("_", strict_key(text, drop_final_schwa=False)).strip("_") or "x"


def entry_from_row(row: Mapping[str, str], *, source: str = "") -> LexiconEntry:
    label = (row.get("label") or "").strip().upper()
    if not label:
        raise LexiconError(f"row without label: {dict(row)}")
    if not is_known(label):
        log.info("registering custom label %s", label)
        register_label(label)

    canonical = normalize_str((row.get("canonical_iast") or "").strip())
    devanagari = normalize_str((row.get("devanagari") or "").strip())
    if canonical.startswith("~") or devanagari.startswith("~"):
        raise LexiconError("use requires_context=yes instead of '~' on canonical/devanagari")
    if not canonical and devanagari:
        canonical = deva_to_iast(devanagari)
    if not devanagari and canonical:
        devanagari = iast_to_deva(canonical) if not is_devanagari(canonical) else canonical
    if not canonical:
        raise LexiconError(f"row without canonical_iast or devanagari: {dict(row)}")

    entry_id = (row.get("id") or "").strip() or f"{label.lower()}:{slugify(canonical)}"
    return LexiconEntry(
        id=entry_id,
        label=label,
        subtype=(row.get("subtype") or "").strip(),
        canonical=canonical,
        devanagari=devanagari,
        synonyms=_split(row.get("synonyms")),
        hindi=_split(row.get("hindi")),
        english=tuple(e.strip() for e in (row.get("english") or "").split(_SPLIT) if e.strip()),
        definition=(row.get("definition") or "").strip(),
        botanical=(row.get("botanical") or "").strip(),
        xrefs=tuple(x.strip() for x in (row.get("xrefs") or "").split(_SPLIT) if x.strip()),
        requires_context=_context(row.get("requires_context")),
        notes=(row.get("notes") or "").strip(),
        source=(row.get("source") or "").strip() or source,
    )


def read_csv(path: str | Path) -> list[LexiconEntry]:
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return _read_rows(csv.DictReader(fh), str(path))


def _read_rows(rows: Iterable[Mapping[str, str]], origin: str) -> list[LexiconEntry]:
    out = []
    for n, row in enumerate(rows, start=2):
        if not any((v or "").strip() for v in row.values()):
            continue
        if (row.get("id") or "").startswith("#"):
            continue
        try:
            out.append(entry_from_row(row, source=Path(origin).stem))
        except LexiconError as exc:
            raise LexiconError(f"{origin}:{n}: {exc}") from exc
    return out


def read_jsonl(path: str | Path) -> list[LexiconEntry]:
    out = []
    with Path(path).open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            row = {k: (_SPLIT.join(v) if isinstance(v, list) else str(v)) for k, v in obj.items()}
            if "canonical" in row and "canonical_iast" not in row:
                row["canonical_iast"] = row.pop("canonical")
            try:
                out.append(entry_from_row(row, source=Path(path).stem))
            except LexiconError as exc:
                raise LexiconError(f"{path}:{n}: {exc}") from exc
    return out


DEFAULT_CATEGORY_MAP = {
    "herbs": "DRAVYA",
    "herb": "DRAVYA",
    "dravya": "DRAVYA",
    "doshas": "DOSHA",
    "dhatus": "DHATU",
    "diseases": "ROGA",
    "symptoms": "SYMPTOM",
    "treatments": "PROCEDURE",
    "formulations": "FORMULATION",
}


def from_term_lists(
    source: str | Path | Mapping[str, list[str]],
    category_map: Mapping[str, str] | None = None,
    *,
    source_name: str = "term-list",
) -> list[LexiconEntry]:
    """Build entries from ``{"category": ["term", ...]}`` (a dict or a JSON file path).

    Every term becomes its own entry; synonyms are not linked. Categories are
    mapped to labels with ``category_map`` (defaults cover herbs/doshas/...),
    and unknown categories are used upper-cased as custom labels.
    """
    if isinstance(source, Mapping):
        data = source
    else:
        data = json.loads(Path(source).read_text(encoding="utf-8"))
    cmap = {k.lower(): v for k, v in (category_map or DEFAULT_CATEGORY_MAP).items()}
    out: list[LexiconEntry] = []
    seen: set[str] = set()
    for category, terms in data.items():
        label = cmap.get(category.lower(), category.upper())
        for term in terms:
            term = normalize_str(term.strip())
            if not term:
                continue
            row = {"label": label}
            row["devanagari" if is_devanagari(term) else "canonical_iast"] = term
            entry = entry_from_row(row, source=source_name)
            if entry.id in seen:
                continue
            seen.add(entry.id)
            out.append(entry)
    return out


def load_path(path: str | Path) -> list[LexiconEntry]:
    """Load one lexicon file, dispatching on extension (.csv, .jsonl, .json term lists)."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".csv":
        return read_csv(p)
    if suffix == ".jsonl":
        return read_jsonl(p)
    if suffix == ".json":
        return from_term_lists(p, source_name=p.stem)
    raise LexiconError(f"unsupported lexicon file type: {p}")


def bundled_paths() -> list[Path]:
    root = resources.files("ayurner.lexicon.data")
    return sorted(Path(str(p)) for p in root.iterdir() if str(p).endswith(".csv"))


def load_bundled() -> list[LexiconEntry]:
    entries: list[LexiconEntry] = []
    for p in bundled_paths():
        entries.extend(read_csv(p))
    return entries


def merge(entries: Iterable[LexiconEntry]) -> list[LexiconEntry]:
    """De-duplicate by id; later entries override earlier ones (user lexicons win)."""
    by_id: dict[str, LexiconEntry] = {}
    for e in entries:
        if e.id in by_id:
            log.debug("lexicon entry %s overridden", e.id)
        by_id[e.id] = e
    return list(by_id.values())
