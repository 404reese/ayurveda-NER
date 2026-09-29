"""Throughput benchmark: tokens/second for one process and for pipe(n_process=N).

Usage:
    python scripts/benchmark.py [--docs 5000] [--processes 4] [-l data/external/nia.csv]
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import ayurner
from ayurner.text.tokenize import tokenize

ROOT = Path(__file__).resolve().parents[1]


def corpus(n: int, seed: int = 0) -> list[str]:
    gold = [
        json.loads(line)["text"]
        for line in (ROOT / "tests/fixtures/gold.jsonl").open(encoding="utf-8")
    ]
    filler = "यह एक सामान्य वाक्य है जिसमें कई साधारण शब्द होते हैं और कुछ शब्द आयुर्वेद से जुड़े होते हैं".split()
    rnd = random.Random(seed)
    docs = []
    for _ in range(n):
        parts = rnd.sample(gold, 3) + [" ".join(rnd.choices(filler, k=20))]
        rnd.shuffle(parts)
        docs.append(" ".join(parts))
    return docs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=5000)
    ap.add_argument("--processes", type=int, default=4)
    ap.add_argument("-l", "--lexicons", nargs="*", default=[])
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    ner = ayurner.load(lexicons=args.lexicons)
    print(f"load: {time.perf_counter() - t0:.2f}s ({len(ner.index)} entries)")

    docs = corpus(args.docs)
    n_tokens = sum(len(tokenize(d)) for d in docs)
    for d in docs[:200]:  # warm per-token caches
        ner(d)

    t0 = time.perf_counter()
    n_ents = sum(len(ner(d)) for d in docs)
    dt = time.perf_counter() - t0
    print(
        f"1 process : {len(docs)} docs, {n_tokens} tokens, {n_ents} entities in {dt:.2f}s "
        f"-> {n_tokens / dt:,.0f} tokens/s, {len(docs) / dt:,.0f} docs/s"
    )

    if args.processes > 1:
        t0 = time.perf_counter()
        n_ents = sum(len(d) for d in ner.pipe(docs, batch_size=128, n_process=args.processes))
        dt = time.perf_counter() - t0
        print(
            f"{args.processes} processes: {n_tokens / dt:,.0f} tokens/s, {len(docs) / dt:,.0f} docs/s "
            "(includes worker start-up)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
