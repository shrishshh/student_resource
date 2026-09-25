"""Indic -> Latin transliteration.

Two layers (runtime order: learned dictionary first, rule-based fallback):

a) :func:`romanize` — rule-based transliterator for the nine Brahmic scripts used in
   the data (Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada,
   Malayalam). Their Unicode blocks share one ISCII-derived layout, so a single
   table keyed by ``codepoint - block_start`` covers all of them; a few
   script-specific extras are layered on top.
b) A token dictionary ``indic_token -> latin_token`` learned from train true pairs
   (``learn`` below), saved to ``artifacts/translit_dict.json``.
c) A component map ``native address component -> canonical state``, learned from
   train true pairs by majority vote, saved to ``artifacts/indic_state_map.json``.

Learning: ``python -m src.translit`` (reads train only; writes artifacts/ and a
section of reports/normalization_report.md input as JSON under artifacts/).
"""

from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
DICT_PATH = ARTIFACTS / "translit_dict.json"
STATE_MAP_PATH = ARTIFACTS / "indic_state_map.json"

# Block start -> script tag.
BLOCKS = {0x0900: "deva", 0x0980: "beng", 0x0A00: "guru", 0x0A80: "gujr", 0x0B00: "orya",
          0x0B80: "taml", 0x0C00: "telu", 0x0C80: "knda", 0x0D00: "mlym"}
# Scripts whose word-final inherent vowel is silent (schwa deletion); in the
# Dravidian scripts a final consonant without virama is pronounced with 'a'.
FINAL_SCHWA_DROP = {"deva", "beng", "guru", "gujr", "orya"}
INDIC_RE = re.compile(r"[ऀ-ൿ‌‍]+")
HAS_INDIC = re.compile(r"[ऀ-ൿ]")

# --- shared offset table ------------------------------------------------------------
INDEPENDENT_VOWELS = {0x04: "a", 0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u",
                      0x0B: "ri", 0x0C: "li", 0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai",
                      0x11: "o", 0x12: "o", 0x13: "o", 0x14: "au", 0x60: "ri", 0x61: "li",
                      0x72: "i", 0x73: "u"}  # 0x72/0x73: Gurmukhi iri/ura bearers
CONSONANTS = {0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n", 0x1A: "ch", 0x1B: "chh",
              0x1C: "j", 0x1D: "jh", 0x1E: "n", 0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh",
              0x23: "n", 0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n",
              0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r",
              0x31: "r", 0x32: "l", 0x33: "l", 0x34: "l", 0x35: "v", 0x36: "sh", 0x37: "sh",
              0x38: "s", 0x39: "h",
              # precomposed nukta letters (Devanagari 0958-095F, Gurmukhi 0A59-0A5E)
              0x58: "q", 0x59: "kh", 0x5A: "g", 0x5B: "z", 0x5C: "r", 0x5D: "rh", 0x5E: "f", 0x5F: "y"}
VOWEL_SIGNS = {0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri", 0x44: "ri",
               0x45: "e", 0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o",
               0x4C: "au", 0x4F: "au", 0x55: "", 0x56: "ai", 0x57: "au", 0x62: "li", 0x63: "li"}
NUKTA_FORMS = {0x15: "q", 0x16: "kh", 0x17: "g", 0x1C: "z", 0x21: "r", 0x22: "rh", 0x2B: "f", 0x2F: "y"}
NASAL = {0x01: "n", 0x02: "n", 0x70: "n"}  # candrabindu, anusvara, Gurmukhi tippi
VIRAMA, NUKTA, VISARGA = 0x4D, 0x3C, 0x03
DIGIT0 = 0x66
# Script-specific dead consonants (no inherent vowel).
DEAD_CONSONANTS = {("beng", 0x4E): "t",            # khanda ta
                   ("mlym", 0x7A): "n", ("mlym", 0x7B): "n", ("mlym", 0x7C): "r",   # chillus
                   ("mlym", 0x7D): "l", ("mlym", 0x7E): "l", ("mlym", 0x7F): "k",
                   ("mlym", 0x4E): "r"}           # dot reph
EXTRA_CONSONANTS = {("orya", 0x71): "w"}
# Whole-sequence rewrites applied before the per-character pass.
PRE_REWRITES = [
    ("റ്റ", "ട്ട"),  # Malayalam റ്റ is pronounced "tt"
    ("ന്റ", "ന്ട"),  # Malayalam ന്റ -> "nt"
    ("ஃப", "஫"),                     # Tamil ஃப -> f (marker char handled below)
]
TAMIL_F = "஫"  # unassigned in Tamil; used as an internal marker for 'f'


def _block(cp: int):
    """(script tag, offset) of an Indic code point, or (None, None)."""
    start = cp & ~0x7F
    tag = BLOCKS.get(start)
    return (tag, cp - start) if tag else (None, None)


@lru_cache(maxsize=500_000)
def romanize(word: str) -> str:
    """Rule-based romanisation of one Indic word (a run of Indic characters).

    Consonants carry an inherent 'a' that a virama removes or a vowel sign
    replaces; the word-final inherent 'a' is dropped in the northern scripts.
    """
    for a, b in PRE_REWRITES:
        word = word.replace(a, b)
    out: list[str] = []
    pending = False  # a consonant with a still-unrealised inherent 'a'
    script = None
    last_cons = None  # offset of the consonant just emitted (for a following nukta)
    for ch in word:
        if ch in "\u200c\u200d":  # ZWNJ / ZWJ
            continue
        if ch == TAMIL_F:
            if pending:
                out.append("a")
            out.append("f")
            pending, script = True, "taml"
            continue
        tag, off = _block(ord(ch))
        if tag is None:
            if pending:
                out.append("a")
                pending = False
            out.append(ch)
            continue
        script = tag
        if (tag, off) in DEAD_CONSONANTS:
            if pending:
                out.append("a")
            out.append(DEAD_CONSONANTS[(tag, off)])
            pending = False
        elif off in CONSONANTS or (tag, off) in EXTRA_CONSONANTS:
            if pending:
                out.append("a")
            out.append(EXTRA_CONSONANTS.get((tag, off)) or CONSONANTS[off])
            pending, last_cons = True, off
            continue
        elif off == NUKTA:
            if last_cons in NUKTA_FORMS:  # e.g. ज़ -> z, फ़ -> f, ड़ -> r
                out[-1] = NUKTA_FORMS[last_cons]
        elif off == VIRAMA:
            pending = False
        elif off in VOWEL_SIGNS:
            out.append(VOWEL_SIGNS[off])
            pending = False
        elif off in INDEPENDENT_VOWELS:
            # a consonant directly followed by an independent vowel loses its inherent
            # 'a' (loanwords such as एलएलपी "elelpi")
            out.append(INDEPENDENT_VOWELS[off])
            pending = False
        elif off in NASAL:
            if pending:
                out.append("a")
            # Malayalam anusvara is a final "m" (keralam)
            out.append("m" if tag == "mlym" and off == 0x02 else "n")
            pending = False
        elif off == VISARGA:
            if tag == "taml":  # aytham not followed by ப: drop
                continue
            if pending:
                out.append("a")
            out.append("h")
            pending = False
        elif DIGIT0 <= off <= DIGIT0 + 9:
            if pending:
                out.append("a")
                pending = False
            out.append(str(off - DIGIT0))
        # anything else in the blocks (avagraha, length marks, signs) is dropped
        if off != NUKTA:
            last_cons = None
    if pending and script not in FINAL_SCHWA_DROP:
        out.append("a")
    return "".join(out)


# --------------------------------------------------------------------------- runtime
class Transliterator:
    """Dictionary-first transliteration of text that may contain Indic words."""

    def __init__(self, token_dict: dict[str, str] | None = None, state_map: dict[str, str] | None = None):
        self.token_dict = token_dict or {}
        self.state_map = state_map or {}

    @classmethod
    def load(cls) -> "Transliterator":
        """Load learned artifacts (empty tables when not learned yet)."""
        td = json.loads(DICT_PATH.read_text(encoding="utf-8")) if DICT_PATH.exists() else {}
        sm = json.loads(STATE_MAP_PATH.read_text(encoding="utf-8")) if STATE_MAP_PATH.exists() else {}
        return cls(td.get("mappings", td), sm.get("mappings", sm))

    def token(self, w: str) -> str:
        """One Indic word -> Latin (dictionary, else rules)."""
        key = normalize_indic_token(w)
        return self.token_dict.get(key) or romanize(key)

    def text(self, s: str) -> str:
        """Replace every Indic run in ``s`` by its romanisation."""
        return INDIC_RE.sub(lambda m: " " + self.token(m.group()) + " ", s)

    def address(self, s: str) -> str:
        """Like :meth:`text`, but whole comma-separated components found in the
        learned state map are replaced by the canonical state name first."""
        if not self.state_map:
            return self.text(s)
        parts = []
        for comp in s.split(","):
            key = normalize_indic_component(comp)
            parts.append(self.state_map[key] if key in self.state_map else self.text(comp))
        return ",".join(parts)


def normalize_indic_token(w: str) -> str:
    """Canonical key for an Indic token (NFKC, no ZWJ/ZWNJ)."""
    return unicodedata.normalize("NFKC", w).replace("‌", "").replace("‍", "")


def normalize_indic_component(c: str) -> str:
    """Canonical key for an address component (stripped, single spaces)."""
    return " ".join(normalize_indic_token(c).split())


# --------------------------------------------------------------------------- learning
MIN_COUNT, MIN_SHARE = 3, 0.6


def _strip_honorifics(tokens: list[str]) -> list[str]:
    """Drop honorific tokens and the 'm s' bigram before alignment."""
    from .resources import HONORIFIC_TOKENS
    out, i = [], 0
    while i < len(tokens):
        if tokens[i] == "m" and i + 1 < len(tokens) and tokens[i + 1] == "s":
            i += 2
            continue
        if tokens[i] not in HONORIFIC_TOKENS:
            out.append(tokens[i])
        i += 1
    return out


def align_name_tokens(noisy: str, s1: str) -> list[tuple[str, str]]:
    """(indic_token, latin_token) alignments for one true pair of cleaned names.

    Positional when both sides have the same number of tokens (after dropping
    honorifics); otherwise each Indic token is matched to the S1 token most
    similar to its rule-based romanisation (rapidfuzz ratio >= 50).
    """
    from rapidfuzz import fuzz, process
    nt, st = _strip_honorifics(noisy.split()), _strip_honorifics(s1.split())
    pos = [i for i, t in enumerate(nt) if HAS_INDIC.search(t)]
    if not pos or not st:
        return []
    if len(nt) == len(st):
        return [(nt[i], st[i]) for i in pos]
    out = []
    for i in pos:
        best = process.extractOne(romanize(nt[i]), st, scorer=fuzz.ratio, score_cutoff=50)
        if best:
            out.append((nt[i], best[0]))
    return out


def _keep(counts: dict, totals: dict) -> dict:
    """Mappings with count >= MIN_COUNT and share >= MIN_SHARE of the source key."""
    best: dict = {}
    for (k, v), c in counts.items():
        if c >= MIN_COUNT and c / totals[k] >= MIN_SHARE and c > best.get(k, (None, 0))[1]:
            best[k] = (v, c)
    return best


def learn() -> dict:
    """Learn the token dictionary and the native-state component map from train pairs."""
    from collections import Counter

    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    from rapidfuzz import fuzz

    from .data import gt_pairs, load_sources, log
    from .normalize import address_features, clean_text
    from .resources import STATE_NAME

    log("translit: loading train")
    src_cols = load_sources("train")
    gt = gt_pairs(src_cols)
    pair_counts, tok_totals = Counter(), Counter()
    comp_votes, comp_totals = Counter(), Counter()
    n_pairs = n_name_pairs = n_addr_pairs = 0
    for src in (2, 3):
        g = gt[gt["src"] == src]
        rec_name = pc.take(src_cols[src]["name"], pa.array(g["idx"].to_numpy()))
        rec_addr = pc.take(src_cols[src]["addr"], pa.array(g["idx"].to_numpy()))
        m_name = pc.fill_null(pc.match_substring_regex(rec_name, r"[\x{0900}-\x{0D7F}]"), False).to_numpy(zero_copy_only=False)
        m_addr = pc.fill_null(pc.match_substring_regex(rec_addr, r"[\x{0900}-\x{0D7F}]"), False).to_numpy(zero_copy_only=False)
        n_pairs += len(g)
        # names: cleaned without transliteration on both sides
        s1_idx = pa.array(g["s1"].to_numpy()[m_name])
        noisy = clean_text(pc.filter(rec_name, pa.array(m_name)), "name", None).to_pylist()
        s1n = clean_text(pc.take(src_cols[1]["name"], s1_idx), "name", None).to_pylist()
        n_name_pairs += len(noisy)
        log(f"translit: S{src} aligning {len(noisy):,} name pairs")
        for a, b in zip(noisy, s1n):
            for t, l in align_name_tokens(a, b):
                t = normalize_indic_token(t)  # runtime keys are NFKC (cleaning ends in NFKD)
                pair_counts[(t, l)] += 1
                tok_totals[t] += 1
        # addresses: native-script components vs the S1 state
        s1a_idx = pa.array(g["s1"].to_numpy()[m_addr])
        raw = pc.utf8_normalize(pc.filter(rec_addr, pa.array(m_addr)), "NFKC").to_pylist()
        s1_clean = clean_text(pc.take(src_cols[1]["addr"], s1a_idx), "addr", None).to_pylist()
        s1_country = pc.take(src_cols[1]["country"], s1a_idx).to_pylist()
        n_addr_pairs += len(raw)
        log(f"translit: S{src} voting {len(raw):,} address pairs")
        for r, s1a, c in zip(raw, s1_clean, s1_country):
            state = address_features(s1a, c)[2]
            if not state:
                continue
            others = [x for x in s1a.split(",") if x and STATE_NAME.get(c, {}).get(state) != x]
            for comp in r.split(","):
                if not HAS_INDIC.search(comp):
                    continue
                key = normalize_indic_component(comp)
                rom = " ".join(romanize(w) for w in key.split())
                # a component that matches another S1 component is a city/district, not the state
                if any(fuzz.ratio(rom, o) >= 80 for o in others):
                    continue
                comp_votes[(key, (c, state))] += 1
                comp_totals[key] += 1
    tok_map = _keep(pair_counts, tok_totals)
    comp_map = _keep(comp_votes, comp_totals)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    DICT_PATH.write_text(json.dumps({
        "min_count": MIN_COUNT, "min_share": MIN_SHARE,
        "mappings": {k: v for k, (v, _) in sorted(tok_map.items())},
        "counts": {k: c for k, (_, c) in sorted(tok_map.items())},
    }, ensure_ascii=False, indent=0), encoding="utf-8")
    STATE_MAP_PATH.write_text(json.dumps({
        "min_count": MIN_COUNT, "min_share": MIN_SHARE,
        "mappings": {k: STATE_NAME[c][s] for k, ((c, s), _) in sorted(comp_map.items())},
        "counts": {k: n for k, (_, n) in sorted(comp_map.items())},
    }, ensure_ascii=False, indent=0), encoding="utf-8")
    stats = {"train_pairs": n_pairs, "indic_name_pairs": n_name_pairs, "indic_addr_pairs": n_addr_pairs,
             "distinct_indic_tokens_aligned": len(tok_totals), "dict_size": len(tok_map),
             "state_map_size": len(comp_map)}
    (ARTIFACTS / "translit_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    log(f"translit: {stats}")
    return stats


if __name__ == "__main__":
    learn()
