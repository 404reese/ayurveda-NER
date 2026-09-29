"""Evaluate ayurner against a gold JSONL file.

Each line: {"text": "...", "entities": [["surface text", "LABEL"], ...], "lang": "hi"}
Gold surfaces are located left to right in the text.

Reports precision / recall / F1 per label for
  * exact:   same span and same label
  * overlap: spans overlap and labels match

Usage:
    python scripts/evaluate.py tests/fixtures/gold.jsonl [-l data/external/nia.csv] [--errors]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict

import ayurner


def locate(text: str, surfaces: list[tuple[str, str]]) -> list[tuple[int, int, str]]:
    out, pos = [], 0
    for surface, label in surfaces:
        i = text.find(surface, pos)
        if i < 0:
            i = text.find(surface)
        if i < 0:
            raise ValueError(f"gold surface {surface!r} not found in {text!r}")
        out.append((i, i + len(surface), label))
        pos = i + len(surface)
    return out


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def evaluate(ner, rows):
    counts = {m: defaultdict(Counter) for m in ("exact", "overlap")}
    errors = []
    by_lang = defaultdict(Counter)
    for row in rows:
        gold = locate(row["text"], [tuple(x) for x in row["entities"]])
        pred = [(e.start, e.end, e.label) for e in ner(row["text"]).entities]
        # exact
        gs, ps = set(gold), set(pred)
        for _s, _e, lab in gs & ps:
            counts["exact"][lab]["tp"] += 1
        for _s, _e, lab in ps - gs:
            counts["exact"][lab]["fp"] += 1
        for _s, _e, lab in gs - ps:
            counts["exact"][lab]["fn"] += 1
        # overlap
        matched_p = set()
        for g in gold:
            hit = next(
                (
                    p
                    for p in pred
                    if p not in matched_p and p[2] == g[2] and p[0] < g[1] and g[0] < p[1]
                ),
                None,
            )
            if hit:
                matched_p.add(hit)
                counts["overlap"][g[2]]["tp"] += 1
            else:
                counts["overlap"][g[2]]["fn"] += 1
        for p in pred:
            if p not in matched_p:
                counts["overlap"][p[2]]["fp"] += 1
        lang = row.get("lang", "?")
        by_lang[lang]["tp"] += len(gs & ps)
        by_lang[lang]["fp"] += len(ps - gs)
        by_lang[lang]["fn"] += len(gs - ps)
        if gs != ps:
            t = row["text"]
            errors.append(
                (
                    t,
                    sorted((t[s:e], lab) for s, e, lab in gs - ps),
                    sorted((t[s:e], lab) for s, e, lab in ps - gs),
                )
            )
    return counts, by_lang, errors


def report(counts, by_lang) -> str:
    lines = []
    for mode in ("exact", "overlap"):
        c = counts[mode]
        lines.append(f"\n{mode.upper():<14}{'P':>7}{'R':>7}{'F1':>7}{'support':>9}")
        tot = Counter()
        for lab in sorted(c):
            tp, fp, fn = c[lab]["tp"], c[lab]["fp"], c[lab]["fn"]
            tot.update(c[lab])
            p, r, f = prf(tp, fp, fn)
            lines.append(f"{lab:<14}{p:>7.2f}{r:>7.2f}{f:>7.2f}{tp + fn:>9}")
        p, r, f = prf(tot["tp"], tot["fp"], tot["fn"])
        lines.append(f"{'micro avg':<14}{p:>7.2f}{r:>7.2f}{f:>7.2f}{tot['tp'] + tot['fn']:>9}")
    lines.append(f"\n{'EXACT by lang':<14}{'P':>7}{'R':>7}{'F1':>7}")
    for lang in sorted(by_lang):
        p, r, f = prf(by_lang[lang]["tp"], by_lang[lang]["fp"], by_lang[lang]["fn"])
        lines.append(f"{lang:<14}{p:>7.2f}{r:>7.2f}{f:>7.2f}")
    return "\n".join(lines)


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("gold")
    ap.add_argument("-l", "--lexicons", nargs="*", default=[])
    ap.add_argument("--errors", action="store_true", help="print every sentence with a mismatch")
    args = ap.parse_args(argv)

    with open(args.gold, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    ner = ayurner.load(lexicons=args.lexicons)
    counts, by_lang, errors = evaluate(ner, rows)
    print(f"{len(rows)} sentences, lexicon_version={ner.lexicon_version}")
    print(report(counts, by_lang))
    if args.errors:
        print("\nMISMATCHES (missed | spurious)")
        for text, missed, spurious in errors:
            print(f"- {text}\n    missed:   {missed}\n    spurious: {spurious}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
