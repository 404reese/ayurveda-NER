"""Convert the NIA / WHO-APW Non-Clinical Terminologies of Ayurveda PDF into an ayurner lexicon CSV.

Usage:
    python scripts/import_nia_terminologies.py TERMINOLOGIES.pdf -o data/external/nia.csv

The PDF is published by the National Institute of Ayurveda (nia.nic.in/pdf/TERMINOLOGIES.pdf).
It carries no explicit licence, so the generated CSV is NOT bundled with the package:
load it with ``ayurner.load(lexicons=["data/external/nia.csv"])`` or AYURNER_LEXICONS.
Labels are provisional (from the discipline code prefix) and should be reviewed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ayurner.lexicon.nia import extract_text, import_entries, split_entries, write_csv


def core_keys() -> dict[tuple[str, ...], set[str]]:
    import ayurner

    idx = ayurner.load(lexicons=[], use_cache=False).index
    keys: dict[tuple[str, ...], set[str]] = {}
    for key, hits in idx.phrases.items():
        keys.setdefault(key, set()).update(idx.entries[h.entry].label for h in hits)
    return keys


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("pdf", help="path to TERMINOLOGIES.pdf")
    ap.add_argument("-o", "--output", default="data/external/nia.csv")
    ap.add_argument(
        "--rejects", help="where to write unparsed rows (default: <output>_rejects.csv)"
    )
    ap.add_argument(
        "--keep-duplicates", action="store_true", help="keep terms already in the core lexicon"
    )
    args = ap.parse_args(argv)

    text = extract_text(args.pdf)
    entries = split_entries(text)
    existing = None if args.keep_duplicates else core_keys()
    rows, rejects, stats = import_entries(entries, existing)

    out = Path(args.output)
    write_csv(rows, out)
    rej = Path(args.rejects) if args.rejects else out.with_name(out.stem + "_rejects.csv")
    write_csv(rejects, rej, columns=("code", "raw"))
    print(stats.summary())
    print(f"wrote {out} and {rej}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
