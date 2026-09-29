"""Regression guard: the gold set must keep a high micro F1.

The gold set is small and was written alongside the lexicon, so this is a
sanity check against regressions, not a measure of real-world accuracy.
"""

import importlib.util
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _evaluate_module():
    spec = importlib.util.spec_from_file_location("evaluate", ROOT / "scripts" / "evaluate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_gold_micro_f1(ner):
    ev = _evaluate_module()
    rows = [
        json.loads(line) for line in (ROOT / "tests/fixtures/gold.jsonl").open(encoding="utf-8")
    ]
    counts, _, _ = ev.evaluate(ner, rows)
    tot = Counter()
    for c in counts["exact"].values():
        tot.update(c)
    _, _, f1 = ev.prf(tot["tp"], tot["fp"], tot["fn"])
    assert f1 >= 0.95, f1
