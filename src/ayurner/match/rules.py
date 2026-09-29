"""Rule layer: disambiguation, composition and context requirements (driven by rules.toml)."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from ..lexicon.index import LexiconIndex
from ..text.phonetic import strict_key
from ..text.tokenize import Token
from .base import BASE_CONFIDENCE, Candidate, Sense
from .gazetteer import token_alts

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib


@dataclass(frozen=True, slots=True)
class PatternElement:
    label: str
    subtype: str = ""
    repeat: bool = False

    @classmethod
    def parse(cls, spec: str) -> PatternElement:
        repeat = spec.endswith("+")
        spec = spec.rstrip("+")
        label, _, subtype = spec.partition(":")
        return cls(label.upper(), subtype, repeat)

    def matches(self, c: Candidate) -> bool:
        if c.final_label != self.label:
            return False
        return not self.subtype or c.final_subtype == self.subtype


@dataclass(frozen=True, slots=True)
class ComposeRule:
    name: str
    pattern: tuple[PatternElement, ...]
    label: str
    subtype: str = ""


@dataclass(frozen=True, slots=True)
class ContextRule:
    cue_labels: frozenset[str]
    cue_words: frozenset[str]
    window: int


def _keys(words) -> frozenset[str]:
    return frozenset(strict_key(w) for w in words if w)


class RuleEngine:
    def __init__(self, config: dict):
        self.compose_rules = tuple(
            ComposeRule(
                r.get("name", ""),
                tuple(PatternElement.parse(p) for p in r["pattern"]),
                r["label"].upper(),
                r.get("subtype", ""),
            )
            for r in config.get("compose", [])
        )
        dis = config.get("disambiguation", {})
        self.dis_window = int(dis.get("window", 6))
        self.priority = {lab: i for i, lab in enumerate(dis.get("priority", []))}
        self.cue_words = {lab.upper(): _keys(ws) for lab, ws in dis.get("cue_words", {}).items()}
        self.label_cues = {
            lab.upper(): frozenset(v) for lab, v in dis.get("label_cues", {}).items()
        }

        ctx = config.get("context", {})
        default_window = int(ctx.get("window", 5))
        self.context_penalty = float(ctx.get("penalty", 0.8))
        self.context_rules: dict[str, ContextRule] = {}
        for key, r in ctx.get("rules", {}).items():
            self.context_rules[key] = ContextRule(
                frozenset(r.get("cue_labels", ["*"])),
                _keys(r.get("cue_words", [])),
                int(r.get("window", default_window)),
            )
        self.context_rules.setdefault(
            "default", ContextRule(frozenset({"*"}), frozenset(), default_window)
        )

    @classmethod
    def from_file(cls, path: str | Path | None = None) -> RuleEngine:
        if path is None:
            raw = resources.files("ayurner.lexicon.data").joinpath("rules.toml").read_bytes()
        else:
            raw = Path(path).read_bytes()
        return cls(tomllib.loads(raw.decode("utf-8")))

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _window_keys(tokens: list[Token], c: Candidate, window: int) -> set[str]:
        seg = tokens[c.tok_start].segment
        lo, hi = max(0, c.tok_start - window), min(len(tokens), c.tok_end + window)
        keys: set[str] = set()
        for t in tokens[lo:hi]:
            if t.segment != seg or c.tok_start <= t.index < c.tok_end:
                continue
            alts = token_alts(t.text)
            keys.update(k for k, _ in alts.strict)
        return keys

    @staticmethod
    def _neighbours(cands: list[Candidate], tokens: list[Token], c: Candidate, window: int):
        seg = tokens[c.tok_start].segment
        for o in cands:
            if o is c or tokens[o.tok_start].segment != seg:
                continue
            gap = max(o.tok_start - c.tok_end, c.tok_start - o.tok_end)
            if gap < window:
                yield o

    # --------------------------------------------------------- disambiguate
    def disambiguate(self, cands: list[Candidate], tokens: list[Token]) -> None:
        for c in cands:
            if c.label or len({s.entry.id for s in c.senses}) < 2:
                continue
            labels = c.labels()
            if len(labels) == 1:
                c.ambiguous = (
                    True  # same label, several entries (e.g. two diseases called "arthritis")
                )
                continue
            keys = self._window_keys(tokens, c, self.dis_window)
            neighbour_labels = [
                o.final_label
                for o in self._neighbours(cands, tokens, c, self.dis_window)
                if len(o.labels()) <= 1 or o.label
            ]
            scores = {}
            for lab in labels:
                score = len(keys & self.cue_words.get(lab, frozenset()))
                score += sum(1 for nl in neighbour_labels if nl in self.label_cues.get(lab, ()))
                scores[lab] = score
            weak_only = {
                lab for lab in labels if all(s.weak for s in c.senses if s.entry.label == lab)
            }
            if weak_only == labels:
                weak_only = set()
            best = max(scores.values())
            # A weak-only sense needs positive cue evidence to beat a strong one.
            order = sorted(
                labels,
                key=lambda lab: (-scores[lab], lab in weak_only, self.priority.get(lab, 99)),
            )
            chosen = order[0]
            if chosen in weak_only and scores[chosen] == 0:
                chosen = next(lab for lab in order if lab not in weak_only)
            no_evidence = best == 0 and len(labels - weak_only) > 1
            tied = best > 0 and sum(1 for lab in labels if scores[lab] == best) > 1
            c.ambiguous = no_evidence or tied
            c.senses = [s for s in c.senses if s.entry.label == chosen] + [
                s for s in c.senses if s.entry.label != chosen
            ]

    # -------------------------------------------------------------- compose
    def _match_rule(self, rule: ComposeRule, seq: list[Candidate], i: int, adjacent) -> int:
        pos = i
        for el in rule.pattern:
            count = 0
            while (
                pos < len(seq)
                and el.matches(seq[pos])
                and (pos == i or adjacent(seq[pos - 1], seq[pos]))
            ):
                pos += 1
                count += 1
                if not el.repeat:
                    break
            if count == 0:
                return 0
        return pos if pos - i >= 2 else 0

    def label_components(self, comps: list[Candidate]) -> tuple[str, str] | None:
        """Label for a compound word whose parts match a compose rule end to end."""
        for rule in self.compose_rules:
            if self._match_rule(rule, comps, 0, lambda a, b: True) == len(comps):
                return rule.label, rule.subtype
        return None

    def compose(
        self, cands: list[Candidate], tokens: list[Token], index: LexiconIndex, text: str
    ) -> list[Candidate]:
        for c in cands:  # compounds first: relabel by compose rules
            if c.components:
                lab = self.label_components(c.components)
                if lab:
                    c.label, c.subtype = lab

        def adjacent(a: Candidate, b: Candidate) -> bool:
            return b.tok_start == a.tok_end and tokens[b.tok_start].joinable

        seq = sorted(cands, key=lambda c: c.tok_start)
        out: list[Candidate] = []
        i = 0
        while i < len(seq):
            best_end, best_rule = 0, None
            for rule in self.compose_rules:
                end = self._match_rule(rule, seq, i, adjacent)
                if end > best_end:
                    best_end, best_rule = end, rule
            if best_rule is None:
                out.append(seq[i])
                i += 1
                continue
            parts = seq[i:best_end]
            out.append(self._merge(parts, best_rule, tokens, index, text))
            i = best_end
        return out

    def _merge(self, parts, rule: ComposeRule, tokens, index: LexiconIndex, text: str) -> Candidate:
        start, end = parts[0].tok_start, parts[-1].tok_end
        conf = min(BASE_CONFIDENCE["compose"], min(p.confidence for p in parts))
        merged = Candidate(
            start,
            end,
            [],
            "compose",
            conf,
            components=list(parts),
            label=rule.label,
            subtype=rule.subtype,
        )
        # Does the composed phrase exist in the lexicon under its own name?
        glued = "".join(t.text for t in tokens[start:end])
        for k, _ in token_alts(glued).strict:
            hits = index.joined.get(k)
            if hits:
                senses = [
                    Sense(index.entries[h.entry], h.weak)
                    for h in hits
                    if index.entries[h.entry].label == rule.label
                ]
                if senses:
                    merged.senses, merged.label, merged.subtype = senses, None, ""
                    break
        return merged

    # -------------------------------------------------------------- context
    def _context_rule(self, c: Candidate) -> ContextRule:
        lab, sub = c.final_label, c.final_subtype
        return (
            self.context_rules.get(f"{lab}:{sub}")
            or self.context_rules.get(lab)
            or self.context_rules["default"]
        )

    def filter_context(self, cands: list[Candidate], tokens: list[Token]) -> list[Candidate]:
        out = []
        strong = [c for c in cands if not (c.weak or c.compose_only)]
        for c in cands:
            if c.compose_only and not c.components:
                continue
            if not c.weak or c.components:
                out.append(c)
                continue
            rule = self._context_rule(c)
            ok = bool(rule.cue_words & self._window_keys(tokens, c, rule.window))
            # Very short weak words (पर, मल, रस) collide with everyday vocabulary:
            # a neighbouring entity is not enough, an explicit cue word is required.
            short = c.length == 1 and len(token_alts(tokens[c.tok_start].text).strict[0][0]) < 4
            if not ok and not short:
                for o in self._neighbours(strong, tokens, c, rule.window):
                    if "*" in rule.cue_labels or o.final_label in rule.cue_labels:
                        ok = True
                        break
            if ok:
                c.confidence *= self.context_penalty
                out.append(c)
        return out
