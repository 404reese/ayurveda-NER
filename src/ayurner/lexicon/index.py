"""Compiled, immutable lexicon index.

Structures (all plain dicts, cheap to pickle and share across processes):

* ``phrases`` / ``loose_phrases``: tuple of per-token keys -> hits. Used for
  single- and multi-word terms ("kālī mirc", "mahānārāyaṇa taila").
* ``prefixes`` / ``loose_prefixes``: every proper prefix of a phrase key, so
  the matcher knows when to keep extending.
* ``joined``: keys with the spaces removed -> hits, for joined spellings
  ("vatapitta", "त्रिफलाचूर्ण") and for compound splitting.
* ``surface``: normalized lower-cased surface form -> entry indexes, used to
  tell exact matches from phonetic ones.
"""

from __future__ import annotations

import hashlib
import itertools
import logging
import os
import pickle
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .._version import __version__
from ..text.normalize import normalize_str
from ..text.phonetic import KEY_VERSION, lexicon_token_keys, loosen, strict_key
from ..text.tokenize import split_words
from .schema import LexiconEntry

log = logging.getLogger(__name__)

MAX_VARIANTS_PER_FORM = 32
EXPANDABLE_LABELS = frozenset({"DOSAGE_FORM"})


@dataclass(frozen=True, slots=True)
class Hit:
    entry: int  # index into LexiconIndex.entries
    weak: bool = False
    english: bool = False


def surface_norm(text: str) -> str:
    return " ".join(split_words(normalize_str(text).lower()))


@dataclass
class LexiconIndex:
    entries: list[LexiconEntry]
    match_english: bool = True
    phrases: dict[tuple[str, ...], tuple[Hit, ...]] = field(default_factory=dict)
    prefixes: set[tuple[str, ...]] = field(default_factory=set)
    loose_phrases: dict[tuple[str, ...], tuple[Hit, ...]] = field(default_factory=dict)
    loose_prefixes: set[tuple[str, ...]] = field(default_factory=set)
    joined: dict[str, tuple[Hit, ...]] = field(default_factory=dict)
    surface: dict[str, frozenset[int]] = field(default_factory=dict)
    by_id: dict[str, int] = field(default_factory=dict)
    max_phrase_len: int = 1
    collisions: list[tuple[str, tuple[str, ...]]] = field(default_factory=list)
    version: str = ""

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, entries: list[LexiconEntry], match_english: bool = True) -> LexiconIndex:
        idx = cls(entries=list(entries), match_english=match_english)
        idx.by_id = {e.id: i for i, e in enumerate(idx.entries)}

        phrases: dict[tuple[str, ...], dict[int, Hit]] = defaultdict(dict)
        loose: dict[tuple[str, ...], dict[int, Hit]] = defaultdict(dict)
        joined: dict[str, dict[int, Hit]] = defaultdict(dict)
        surface: dict[str, set[int]] = defaultdict(set)

        forms_by_entry = [e.surface_forms(match_english) for e in idx.entries]

        # Token-level synonym expansion for dosage forms: "taila" in
        # "mahānārāyaṇa taila" also matches "tel", "oil", "तेल" ...
        expansion: dict[str, set[str]] = {}
        for e, forms in zip(idx.entries, forms_by_entry, strict=True):
            if e.label not in EXPANDABLE_LABELS:
                continue
            keys: set[str] = set()
            for f in forms:
                words = split_words(normalize_str(f.text))
                if len(words) == 1 and not f.weak:
                    keys |= lexicon_token_keys(words[0])
            for k in keys:
                expansion.setdefault(k, set()).update(keys)

        def put(table, key, hit: Hit) -> None:
            slot = table[key]
            prev = slot.get(hit.entry)
            # Keep the strongest flag combination for this entry.
            if prev is None or (prev.weak and not hit.weak):
                slot[hit.entry] = hit

        for ei, forms in enumerate(forms_by_entry):
            for f in forms:
                norm = normalize_str(f.text)
                words = split_words(norm)
                if not words:
                    continue
                surface[" ".join(w.lower() for w in words)].add(ei)
                hit = Hit(ei, f.weak, f.english)
                per_token = []
                for w in words:
                    keys = set(lexicon_token_keys(w)) if not f.english else {strict_key(w)}
                    if len(words) > 1:
                        for k in list(keys):
                            keys |= expansion.get(k, set())
                    per_token.append(sorted(keys))
                for combo in itertools.islice(itertools.product(*per_token), MAX_VARIANTS_PER_FORM):
                    put(phrases, combo, hit)
                    if not f.english:
                        put(loose, tuple(loosen(k) for k in combo), hit)
                idx.max_phrase_len = max(idx.max_phrase_len, len(words))
                if not f.english:
                    glued = "".join(words)
                    jkeys = lexicon_token_keys(glued) | {strict_key(glued, drop_final_schwa=False)}
                    for w in words:  # also the no-drop key of each single-word form
                        if len(words) == 1:
                            jkeys.add(strict_key(w, drop_final_schwa=False))
                    for k in jkeys:
                        if len(k) >= 3:
                            put(joined, k, hit)

        idx.phrases = {k: tuple(v.values()) for k, v in phrases.items()}
        idx.loose_phrases = {k: tuple(v.values()) for k, v in loose.items()}
        idx.joined = {k: tuple(v.values()) for k, v in joined.items()}
        idx.surface = {k: frozenset(v) for k, v in surface.items()}
        idx.prefixes = {k[:i] for k in idx.phrases for i in range(1, len(k))}
        idx.loose_prefixes = {k[:i] for k in idx.loose_phrases for i in range(1, len(k))}
        idx.collisions = idx._find_collisions()
        return idx

    def _find_collisions(self) -> list[tuple[str, tuple[str, ...]]]:
        out = []
        for key, hits in self.phrases.items():
            strong = [h for h in hits if not h.weak]
            labels = {self.entries[h.entry].label for h in strong}
            if len(labels) > 1:
                out.append((" ".join(key), tuple(sorted(self.entries[h.entry].id for h in strong))))
        return sorted(out)

    # ----------------------------------------------------------------- access
    def entry(self, i: int) -> LexiconEntry:
        return self.entries[i]

    def get(self, entry_id: str) -> LexiconEntry | None:
        i = self.by_id.get(entry_id)
        return None if i is None else self.entries[i]

    def collision_report(self) -> str:
        if not self.collisions:
            return "no cross-label key collisions"
        lines = [f"{len(self.collisions)} keys map to entries with different labels:"]
        lines += [f"  {k!r}: {', '.join(ids)}" for k, ids in self.collisions]
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.entries)


# ------------------------------------------------------------------- caching


def _cache_dir() -> Path:
    env = os.environ.get("AYURNER_CACHE_DIR")
    if env:
        return Path(env)
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "ayurner" / "Cache"
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "ayurner"


def fingerprint(sources: list[bytes], match_english: bool) -> str:
    h = hashlib.sha256()
    h.update(f"{__version__}|{KEY_VERSION}|{match_english}".encode())
    for s in sources:
        h.update(hashlib.sha256(s).digest())
    return h.hexdigest()


def build_cached(
    entries_loader, source_blobs: list[bytes], match_english: bool, use_cache: bool = True
) -> LexiconIndex:
    """Build an index, reusing a pickled copy keyed by the lexicon content hash."""
    fp = fingerprint(source_blobs, match_english)
    path = _cache_dir() / f"index-{fp[:24]}.pkl"
    if use_cache and path.exists():
        try:
            with path.open("rb") as fh:
                idx = pickle.load(fh)
            if isinstance(idx, LexiconIndex) and idx.version == fp[:12]:
                return idx
        except Exception as exc:  # corrupt / incompatible cache: rebuild
            log.warning("ignoring unreadable index cache %s: %s", path, exc)
    idx = LexiconIndex.build(entries_loader(), match_english=match_english)
    idx.version = fp[:12]
    if use_cache:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
            with os.fdopen(fd, "wb") as fh:
                pickle.dump(idx, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, path)
        except OSError as exc:
            log.warning("could not write index cache %s: %s", path, exc)
    return idx
