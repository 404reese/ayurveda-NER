"""Importer for the NIA / WHO-APW "Standardization of Non-Clinical Terminologies of Ayurveda" PDF.

Each entry in the PDF is ``CODE  Devanagari  Roman(IAST)  English meaning``,
e.g. ``SK0186 शुक्र धातु śukra dhātu The seventh Dhātu...``.

The Devanagari column was produced with a legacy font mapping and extracts as
garbage (आचाय for आचार), but the IAST column is clean Unicode. We therefore
keep the IAST, *regenerate* the Devanagari from it, and use the Devanagari
column only to count how many words the term has.

Labels are provisional, derived from the discipline code prefix plus a few
keyword heuristics. Review them before relying on them.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from ..text.tokenize import is_devanagari
from ..text.translit import iast_to_deva
from .schema import CSV_COLUMNS

SOURCE = "NIA-WHO-APW"

DISCIPLINES = {
    "AF": "Fundamental terms (clinical features)",
    "BP": "Basic principles",
    "SR": "Sharira Rachana",
    "SK": "Sharira Kriya",
    "SV": "Swastha Vritta",
    "DG": "Dravya Guna",
    "RS": "Rasa Shastra",
    "BK": "Bhaishajya Kalpana",
    "AT": "Agada Tantra",
}

_ENTRY_RE = re.compile(r"(?m)^\s*([A-Z]{2}\d{4})\s+")
_PAGE_NUMBER_RE = re.compile(r"(?m)^\s*\d{1,3}\s*$")
_HEADER_RE = re.compile(r"(?m)^\s*AN Code Devanagari Roman English Meaning\s*$")
_IAST_WORD_RE = re.compile(r"^[a-zāīūṛṝḷḹṃṁḥṅñṭḍṇśṣ'’()/\-]+$")
# Look-alike characters used in the PDF instead of IAST letters.
_CHAR_FIXES = {"ῡ": "ū", "ē": "e", "ō": "o", "¡": "ā"}
_IAST_ONLY_CHARS = set("āīūṛṝḷḹṃṁḥṅñṭḍṇśṣ")
_BINOMIAL_RE = re.compile(r"\b[A-Z][a-z]+ [a-z]{3,}\b")

_DOSAGE_WORDS = (
    "powder",
    "decoction",
    "paste",
    "pill",
    "tablet",
    "oil",
    "ghee",
    "juice",
    "infusion",
    "linctus",
    "confection",
    "fermented",
    "preparation",
    "formulation",
    "ash",
    "calx",
)
_PROCESS_WORDS = (
    "process",
    "procedure",
    "method",
    "technique",
    "heating",
    "boiling",
    "trituration",
    "incineration",
    "purification",
    "melting",
    "grinding",
    "washing",
    "roasting",
)
_MINERAL_WORDS = (
    "mercury",
    "sulphur",
    "sulfur",
    "mica",
    "ore",
    "metal",
    "gem",
    "stone",
    "salt",
    "copper",
    "iron",
    "gold",
    "silver",
    "zinc",
    "lead",
    "tin",
    "arsenic",
    "pyrite",
)


@dataclass
class ImportStats:
    entries: int = 0
    written: int = 0
    rejected: int = 0
    skipped_existing: int = 0
    by_label: Counter = field(default_factory=Counter)

    def summary(self) -> str:
        labels = ", ".join(f"{k}={v}" for k, v in self.by_label.most_common())
        return (
            f"{self.entries} entries parsed: {self.written} written, {self.rejected} rejected, "
            f"{self.skipped_existing} already in core lexicon\n  labels: {labels}"
        )


def extract_text(pdf_path: str | Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise ImportError("the NIA importer needs pypdf: pip install 'ayurner[pdf]'") from exc
    reader = PdfReader(str(pdf_path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def split_entries(text: str) -> list[tuple[str, str]]:
    """Return (code, body) pairs; body joins the entry's wrapped lines."""
    text = _HEADER_RE.sub("", text)
    text = _PAGE_NUMBER_RE.sub("", text)
    matches = list(_ENTRY_RE.finditer(text))
    out = []
    for m, nxt in zip(matches, [*matches[1:], None], strict=True):
        body = text[m.end() : nxt.start() if nxt else len(text)]
        out.append((m.group(1), " ".join(body.split())))
    return out


def _fix_iast(word: str) -> str:
    word = unicodedata.normalize("NFC", word).replace("’", "'")
    return word.replace("ṁ", "ṃ")


def _is_iast_word(word: str) -> bool:
    return bool(_IAST_WORD_RE.match(unicodedata.normalize("NFC", word)))


def parse_body(body: str) -> tuple[str, str] | None:
    """Split an entry body into (iast_term, english). None if it cannot be parsed."""
    for a, b in _CHAR_FIXES.items():
        body = body.replace(a, b)
    body = re.sub(r"\s*/\s*", "/", body)  # "aviṣa/ nirviṣ" -> "aviṣa/nirviṣ"
    words = body.split()
    deva = 0
    while deva < len(words) and is_devanagari(words[deva]):
        deva += 1
    rest = words[deva:]
    if deva == 0 or not rest:
        return None
    n = min(deva, len(rest))
    # Shrink: an English word (capitalized, or not IAST-shaped) ends the term early.
    for i in range(n):
        w = rest[i]
        if i > 0 and (w[:1].isupper() or not _is_iast_word(w.lower())):
            n = i
            break
    # Extend: following lowercase words carrying IAST-only letters still belong to the term.
    while (
        n < len(rest)
        and rest[n][:1].islower()
        and _is_iast_word(rest[n])
        and set(rest[n]) & _IAST_ONLY_CHARS
    ):
        n += 1
    term_words = [
        _fix_iast(w.lower() if i == len(rest[:n]) - 1 else w) for i, w in enumerate(rest[:n])
    ]
    term = " ".join(w.lower() for w in term_words).strip()
    english = " ".join(rest[n:]).strip()
    if not term or not english or not all(_is_iast_word(w) for w in term.split()):
        return None
    return term, english


def provisional_label(code: str, iast: str, english: str) -> tuple[str, str]:
    prefix = code[:2]
    en = english.lower()
    if prefix == "AF":
        return "SYMPTOM", ""
    if prefix == "SR":
        return "ANATOMY", ""
    if prefix == "SK":
        if "srota" in iast or "srotas" in iast:
            return "ANATOMY", "srotas"
        if "upadhātu" in en or "upadhatu" in en:
            return "DHATU", "upadhatu"
        if "dhātu" in iast:
            return "DHATU", ""
        if re.search(r"\b(vāta|pitta|kapha|doṣa)\b", iast):
            return "DOSHA", ""
        if re.search(r"\bmala\b", iast):
            return "MALA", ""
        return "CONCEPT", "physiology"
    if prefix == "DG":
        if re.match(r"(substances?|drugs?|herbs?)\s+(which|that)", en):
            return "KARMA", ""
        if "combination of" in en or "group of" in en:
            return "FORMULATION", "gana"
        if _BINOMIAL_RE.search(english):
            return "DRAVYA", "herb"
        return "CONCEPT", "dravyaguna"
    if prefix == "BK":
        if any(w in en for w in _PROCESS_WORDS):
            return "PROCEDURE", "pharmaceutical"
        if any(re.search(rf"\b{w}\b", en) for w in _DOSAGE_WORDS):
            return "DOSAGE_FORM", ""
        return "CONCEPT", "pharmaceutics"
    if prefix == "RS":
        if any(w in en for w in _PROCESS_WORDS):
            return "PROCEDURE", "rasashastra"
        if any(re.search(rf"\b{w}\b", en) for w in _MINERAL_WORDS):
            return "DRAVYA", "mineral"
        return "CONCEPT", "rasashastra"
    if prefix == "AT":
        return "CONCEPT", "toxicology"
    return "CONCEPT", DISCIPLINES.get(prefix, "").split()[
        0
    ].lower() if prefix in DISCIPLINES else ""


def _short_english(english: str) -> str:
    """A short gloss (≤4 words) usable as an English synonym, else ''."""
    first = re.split(r"[;.:]|\s[-–]\s", english, maxsplit=1)[0].strip().strip(",")
    if 0 < len(first.split()) <= 4 and not re.search(r"[()/]", first):
        return first
    return ""


def to_row(code: str, iast: str, english: str) -> dict[str, str]:
    # "ghuṭikā/ghuṇṭikā" lists alternative spellings: first is canonical, rest are synonyms.
    variants = [" ".join(v.split()) for v in iast.split("/") if v.strip()]
    iast, synonyms = variants[0], variants[1:]
    label, subtype = provisional_label(code, iast, english)
    single = " " not in iast
    return {
        "id": f"nia:{code}",
        "label": label,
        "subtype": subtype,
        "canonical_iast": iast,
        "devanagari": iast_to_deva(iast),
        "synonyms": "|".join(synonyms),
        "hindi": "",
        "english": _short_english(english),
        "definition": english,
        "botanical": "",
        "xrefs": f"nia:{code}",
        # Single-word concept terms are often ordinary words (deśa = place); require context.
        "requires_context": "yes" if (label == "CONCEPT" and single) else "",
        "notes": f"provisional label from {DISCIPLINES.get(code[:2], code[:2])}",
        "source": SOURCE,
    }


def import_entries(
    entries: Iterable[tuple[str, str]], existing_keys: dict[tuple[str, ...], set[str]] | None = None
) -> tuple[list[dict], list[dict], ImportStats]:
    """Parse (code, body) pairs into lexicon rows and rejects.

    ``existing_keys`` maps phrase keys of an existing lexicon to its labels;
    NIA terms that duplicate an existing term with the same label are skipped.
    """
    from ..text.phonetic import strict_key
    from ..text.tokenize import split_words

    stats = ImportStats()
    rows, rejects = [], []
    for code, body in entries:
        stats.entries += 1
        parsed = parse_body(body)
        if not parsed:
            stats.rejected += 1
            rejects.append({"code": code, "raw": body})
            continue
        row = to_row(code, *parsed)
        if existing_keys is not None:
            key = tuple(strict_key(w) for w in split_words(row["canonical_iast"]))
            if row["label"] in existing_keys.get(key, ()):
                stats.skipped_existing += 1
                continue
        rows.append(row)
        stats.written += 1
        stats.by_label[row["label"]] += 1
    return rows, rejects, stats


def write_csv(rows: list[dict], path: str | Path, columns=CSV_COLUMNS) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns))
        w.writeheader()
        w.writerows(rows)
