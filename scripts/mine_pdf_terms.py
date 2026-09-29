"""Mine candidate Ayurvedic terms from PDFs / text files for manual review.

Extracts the text layer, finds frequent words and 2-3 word phrases that the
current lexicon does NOT recognize, and writes them to a CSV with a context
snippet and an empty ``label`` column. Fill in the labels (and canonical IAST
if you like), keep the useful rows, and load the file as an extra lexicon.

Usage:
    python scripts/mine_pdf_terms.py book.pdf notes.txt -o candidates.csv [--min-freq 3]

Scanned PDFs and PDFs typeset with legacy Hindi fonts (Kruti Dev etc.) have no
usable text layer; the script warns about them. Such files need OCR first
(e.g. Tesseract with the ``hin``/``san`` models), which is outside this script.
Only mine documents you have the right to use.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import ayurner
from ayurner.text.normalize import normalize_str
from ayurner.text.phonetic import is_casual_latin, strict_key
from ayurner.text.tokenize import tokenize
from ayurner.text.translit import transliterate

# Very common Hindi / Sanskrit function words to ignore as candidates.
STOPWORDS = set(
    """है हैं था थे थी और या तथा के का की को में से पर भी ही यह वह इस उस एक कि जो तो न नहीं
    लिए साथ द्वारा होता होती होते किया करें करते करना जाता जाती जाते गया गई हो रहा रही अपने
    च वा तु हि अपि इति एव सः सा तत् तस्य यस्य यत् न तथा च यथा अथ""".split()
)

_DEVA_CHARS = re.compile(r"[ऀ-ॿ]")
_MATRA_AT_START = re.compile(r"^[ा-ौॢॣ]")
# Characters that are rare in real Hindi/Sanskrit but common in legacy-font mis-mappings.
_LEGACY_ARTIFACTS = set("ऩऱॊॆऻऺऄ")


def read_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from ayurner.lexicon.nia import extract_text

        return extract_text(path)
    return path.read_text(encoding="utf-8", errors="replace")


def garble_score(text: str) -> float:
    """Share of Devanagari words that look mis-mapped (legacy-font artifacts, leading vowel signs)."""
    words = [w for w in text.split() if _DEVA_CHARS.search(w)]
    if not words:
        return 0.0
    bad = sum(1 for w in words if _MATRA_AT_START.match(w) or set(w) & _LEGACY_ARTIFACTS)
    return bad / len(words)


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("files", nargs="+")
    ap.add_argument("-o", "--output", default="candidates.csv")
    ap.add_argument("--min-freq", type=int, default=3)
    ap.add_argument("--max-ngram", type=int, default=3)
    ap.add_argument("-l", "--lexicons", nargs="*", default=[])
    ap.add_argument(
        "--latin", action="store_true", help="also mine plain-ASCII words (English, Hinglish)"
    )
    args = ap.parse_args(argv)

    ner = ayurner.load(lexicons=args.lexicons)
    counts: Counter = Counter()
    contexts: dict[tuple, str] = {}
    sources: dict[tuple, set] = defaultdict(set)

    for f in map(Path, args.files):
        text = normalize_str(read_text(f))
        g = garble_score(text)
        if g > 0.05:
            print(
                f"warning: {f.name}: {g:.0%} of Devanagari words look mis-mapped; "
                "the text layer is probably garbled (legacy font) - consider OCR",
                file=sys.stderr,
            )
        if not text.strip():
            print(f"warning: {f.name}: no text layer (scanned PDF?) - needs OCR", file=sys.stderr)
            continue
        for segment in re.split(r"[।॥.\n]+", text):
            doc = ner(segment)
            covered = set()
            for e in doc.entities:
                covered.update(range(e.start, e.end))
            toks = [
                t
                for t in tokenize(segment)
                if len(t.text) > 2
                and (args.latin or t.is_devanagari or not is_casual_latin(t.text))
            ]
            for n in range(1, args.max_ngram + 1):
                for i in range(len(toks) - n + 1):
                    gram = toks[i : i + n]
                    words = [t.text for t in gram]
                    if any(w in STOPWORDS for w in words) or any(t.start in covered for t in gram):
                        continue
                    if n > 1 and not all(t.joinable for t in gram[1:]):
                        continue
                    key = tuple(strict_key(w) for w in words)
                    counts[key] += 1
                    sources[key].add(f.name)
                    if key not in contexts:
                        s = max(0, gram[0].start - 40)
                        contexts[key] = (" ".join(words), segment[s : gram[-1].end + 40].strip())

    rows = []
    for key, freq in counts.most_common():
        if freq < args.min_freq:
            break
        surface, ctx = contexts[key]
        rows.append(
            {
                "term": surface,
                "iast": transliterate(surface, to="iast"),
                "frequency": freq,
                "files": ";".join(sorted(sources[key])),
                "context": ctx,
                "label": "",
            }
        )
    with open(args.output, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["term", "iast", "frequency", "files", "context", "label"]
        )
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} candidates (freq >= {args.min_freq}) written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
