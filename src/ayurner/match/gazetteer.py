"""Dictionary recognizer: phrase-trie walk, joined spellings and compound splitting."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from ..lexicon.index import Hit, LexiconIndex, surface_norm
from ..text.phonetic import VOWELS, drop_schwa, is_casual_latin, loosen, strict_key
from ..text.stem import is_suffix, stem_candidates
from ..text.tokenize import Token
from .base import BASE_CONFIDENCE, Candidate, MatchContext, Sense

_MT_RANK = {"phonetic": 0, "stem": 1, "loose": 2}


@dataclass(frozen=True, slots=True)
class TokenAlts:
    full: str  # strict key, no final-schwa drop
    strict: tuple[tuple[str, str], ...]  # (key, match_type)
    loose: tuple[tuple[str, str], ...]


@lru_cache(maxsize=200_000)
def token_alts(text: str) -> TokenAlts:
    full = strict_key(text, drop_final_schwa=False)
    drop = drop_schwa(full)
    strict: list[tuple[str, str]] = [(drop, "phonetic")]
    if full != drop:
        strict.append((full, "phonetic"))
    for s in stem_candidates(full):
        if s != drop:
            strict.append((s, "stem"))
    loose: list[tuple[str, str]] = []
    if is_casual_latin(text):
        seen = set()
        for k, _ in strict:
            lk = loosen(k)
            if lk not in seen:
                seen.add(lk)
                loose.append((lk, "loose"))
    return TokenAlts(full, tuple(strict), tuple(loose))


def _senses(index: LexiconIndex, hits: tuple[Hit, ...], latin_ok: bool) -> list[Sense]:
    hits = tuple(h for h in hits if latin_ok or not h.english)
    # Strong senses first; weak ones stay as alternatives that context may promote.
    ordered = [h for h in hits if not h.weak] + [h for h in hits if h.weak]
    return [Sense(index.entries[h.entry], h.weak) for h in ordered]


class GazetteerRecognizer:
    """Finds lexicon terms by walking the phrase trie over per-token key alternatives."""

    def __init__(self, max_states: int = 64, min_compound_len: int = 6, min_part_len: int = 3):
        self.max_states = max_states
        self.min_compound_len = min_compound_len
        self.min_part_len = min_part_len

    # ------------------------------------------------------------ public API
    def recognize(self, ctx: MatchContext) -> list[Candidate]:
        tokens, index = ctx.tokens, ctx.index
        alts = [token_alts(t.text) for t in tokens]
        out: list[Candidate] = []
        for i in range(len(tokens)):
            found = self._walk(ctx, alts, i, loose=False)
            if not found and alts[i].loose:
                found = self._walk(ctx, alts, i, loose=True)
            out.extend(found)

        covered = set()
        for c in out:
            covered.update(range(c.tok_start, c.tok_end))
        for i, tok in enumerate(tokens):
            if i in covered:
                continue
            cand = self._joined(index, tok, alts[i], i) or self._compound(index, tok, alts[i], i)
            if cand:
                out.append(cand)
        return out

    # --------------------------------------------------------------- phrases
    def _walk(
        self, ctx: MatchContext, alts: list[TokenAlts], i: int, loose: bool
    ) -> list[Candidate]:
        index, tokens = ctx.index, ctx.tokens
        phrases = index.loose_phrases if loose else index.phrases
        prefixes = index.loose_prefixes if loose else index.prefixes
        results: list[Candidate] = []
        states: list[tuple[tuple[str, ...], str]] = [((), "phonetic")]
        j = i
        while j < len(tokens) and j - i < index.max_phrase_len:
            if j > i and (not tokens[j].joinable or tokens[j].segment != tokens[i].segment):
                break
            options = alts[j].loose if loose else alts[j].strict
            new_states: dict[tuple[str, ...], str] = {}
            for keys, mt in states:
                for k, kmt in options:
                    nk = keys + (k,)
                    if nk in phrases or nk in prefixes:
                        worst = max(mt, kmt, key=lambda m: _MT_RANK.get(m, 0))
                        if nk not in new_states or _MT_RANK[worst] < _MT_RANK[new_states[nk]]:
                            new_states[nk] = worst
            if not new_states:
                break
            states = list(new_states.items())[: self.max_states]
            best: tuple[tuple[str, ...], str] | None = None
            for keys, mt in states:
                if keys in phrases and (best is None or _MT_RANK[mt] < _MT_RANK[best[1]]):
                    best = (keys, mt)
            if best:
                cand = self._make(ctx, i, j + 1, phrases[best[0]], best[1], best[0])
                if cand:
                    results.append(cand)
            j += 1
        return results

    def _make(self, ctx, start, end, hits, match_type, keys) -> Candidate | None:
        tokens, index = ctx.tokens, ctx.index
        latin_ok = not any(t.is_devanagari for t in tokens[start:end])
        senses = _senses(index, hits, latin_ok)
        if not senses:
            return None
        text = ctx.normalized.text[tokens[start].start : tokens[end - 1].end]
        exact_ids = index.surface.get(surface_norm(text), frozenset())
        exact = [s for s in senses if index.by_id[s.entry.id] in exact_ids]
        if exact:
            senses = exact + [s for s in senses if s not in exact]
            match_type = "exact"
        elif end - start == 1 and len(keys[0]) < 3:
            return None  # very short keys (e.g. "el") must match a surface form exactly
        return Candidate(start, end, senses, match_type, BASE_CONFIDENCE[match_type])

    # ------------------------------------------------------- joined spellings
    def _joined(self, index: LexiconIndex, tok: Token, alts: TokenAlts, i: int) -> Candidate | None:
        latin_ok = not tok.is_devanagari
        for k, mt in alts.strict:
            hits = index.joined.get(k)
            if hits:
                senses = _senses(index, hits, latin_ok)
                if senses:
                    return Candidate(i, i + 1, senses, mt, BASE_CONFIDENCE[mt])
        return None

    # ----------------------------------------------------- compound splitting
    def _compound(
        self, index: LexiconIndex, tok: Token, alts: TokenAlts, i: int
    ) -> Candidate | None:
        key = alts.full
        if len(key) < self.min_compound_len:
            return None
        parts = split_compound(index, key, self.min_part_len)
        if not parts:
            return None
        comps: list[Candidate] = []
        for hits in parts:
            senses = _senses(index, hits, latin_ok=False)
            if not senses:
                return None
            comps.append(Candidate(i, i + 1, senses, "compound", BASE_CONFIDENCE["compound"]))
        if all(c.weak or c.compose_only for c in comps):
            return None
        head = comps[-1]
        return Candidate(
            i,
            i + 1,
            [],
            "compound",
            BASE_CONFIDENCE["compound"],
            components=comps,
            label=head.final_label,
            subtype=head.final_subtype,
        )


def split_compound(
    index: LexiconIndex, key: str, min_part: int = 3
) -> list[tuple[Hit, ...]] | None:
    """Segment a folded key into lexicon keys (fewest parts, ≥2 parts).

    Allows a shared vowel at a boundary (dīrgha sandhi: aśoka + ariṣṭa →
    aśokāriṣṭa) and a trailing case ending (vāta + roga + -eṣu).
    """
    n = len(key)
    joined = index.joined
    # best[p] = (n_parts, -chars_matched, parts) for a segmentation covering key[:p]
    best: list[tuple[int, int, list[tuple[int, int, tuple[Hit, ...]]]] | None] = [None] * (n + 1)
    best[0] = (0, 0, [])
    for p in range(n):
        state = best[p]
        if state is None:
            continue
        starts = [p]
        if p > 0 and state[2] and key[p - 1] in VOWELS:
            starts.append(p - 1)
        for s in starts:
            for e in range(s + min_part, n + 1):
                hits = joined.get(key[s:e])
                if not hits:
                    continue
                cand = (state[0] + 1, state[1] - (e - s), state[2] + [(s, e, hits)])
                cur = best[e]
                if cur is None or cand[:2] < cur[:2]:
                    best[e] = cand
    finals = []
    if best[n] is not None:
        finals.append(best[n])
    for p in range(1, n):
        if best[p] is not None and best[p][0] >= 2 and is_suffix(key[p:]):
            finals.append(best[p])
    finals = [f for f in finals if f[0] >= 2]
    if not finals:
        return None
    chosen = min(finals, key=lambda f: f[:2])
    if max(e - s for s, e, _ in chosen[2]) < 4:
        return None
    return [hits for _, _, hits in chosen[2]]
