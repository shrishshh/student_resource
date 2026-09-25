"""Record normalisation -> cache/{split}_records.parquet.

Usage (from code/business_entity_resolution/):
    python -m src.normalize --split train
    python -m src.normalize --split test

Pipeline per field
------------------
1. Generic cleaning (:func:`clean_text`, vectorised with pyarrow; Python only on
   the subsets that need it): mojibake fix -> NFKC -> Indic transliteration ->
   NFKD + drop U+0300-U+036F -> ligatures -> lower-case -> '&'/'+' -> "and" ->
   collapse dotted single letters (l.l.c. -> llc) -> French elisions -> drop other
   apostrophes -> punctuation -> space -> collapse spaces.
2. Names (:class:`NameProcessor`): name_clean, name_core (honorifics, handles, tags,
   phone numbers, web parts, legal forms, stop words and repeated tokens removed),
   name_alt (the real name after a trade/former-name marker), glued-word
   segmentation against the split+country S1 vocabulary, name_concat, name_phon.
3. Addresses (:func:`address_features`): placeholders dropped, state detected and
   canonicalised (country-keyed tables), abbreviations expanded, numbers extracted.

Every rule list lives in ``resources.py``; the learned transliteration artifacts come
from ``python -m src.translit``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter
from functools import lru_cache

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from . import resources as R
from .data import CACHE, LARGE, load_sources, log, peak_rss
from .translit import HAS_INDIC, Transliterator

CHUNK = 500_000

# ----------------------------------------------------------------- generic cleaning
_DOTTED_RE2 = r"(^|[^\p{L}\p{N}.])\p{L}\.\p{L}(\.|[^\p{L}\p{N}]|$)"
_DOTTED_PY = re.compile(r"(?<![\w.])((?:[^\W\d_]\.)+[^\W\d_])\.?(?!\w)")
_PHONE_TAG_RE2 = r"(?i)\(?\bid\s*:?\s*\d+\)?|#\s*\d+|\+?(?:\d[ ().-]{0,2}){6,}\d"


def _apply_subset(arr: pa.Array, re2_pattern: str, fn) -> pa.Array:
    """Apply a Python function only to the rows matching an RE2 pattern."""
    mask = pc.fill_null(pc.match_substring_regex(arr, re2_pattern), False)
    n = pc.sum(mask).as_py() or 0
    if n == 0:
        return arr
    vals = pc.filter(arr, mask).to_pylist()
    return pc.replace_with_mask(arr, mask, pa.array([fn(v) for v in vals], LARGE))


def _collapse_dotted(s: str) -> str:
    """'l.l.c.' -> 'llc', 'e.u.r.l.' -> 'eurl', 'p.c' -> 'pc'."""
    return _DOTTED_PY.sub(lambda m: m.group(1).replace(".", ""), s)


def clean_text(arr: pa.Array, mode: str, tr: Transliterator | None) -> pa.Array:
    """Generic cleaning of a name ("name") or address ("addr") column.

    Address mode keeps commas (component separators), removes '#' without inserting
    a space (d##85 -> d85) and blanks "<null>" / "n/a" placeholders.
    ``tr=None`` keeps Indic text untouched (used when learning the dictionary).
    """
    x = pc.replace_substring_regex(arr, r"[\x{00C2}\x{00E2}]?[\x{0080}-\x{009F}]", "")  # mojibake
    x = pc.replace_substring_regex(x, r"[\x{200B}-\x{200D}\x{2060}\x{FEFF}]", "")  # zero-width chars
    x = pc.utf8_normalize(x, "NFKC")
    if tr is not None:
        x = _apply_subset(x, r"[\x{0900}-\x{0D7F}]", tr.address if mode == "addr" else tr.text)
    x = pc.utf8_normalize(x, "NFKD")
    x = pc.replace_substring_regex(x, r"[\x{0300}-\x{036f}]", "")
    for a, b in (("œ", "oe"), ("Œ", "oe"), ("æ", "ae"), ("Æ", "ae"), ("ß", "ss")):
        x = pc.replace_substring(x, a, b)
    x = pc.utf8_lower(x)
    x = pc.replace_substring_regex(x, r"[&+]", " and ")
    x = _apply_subset(x, _DOTTED_RE2, _collapse_dotted)
    x = pc.replace_substring_regex(x, r"(^|[^\p{L}\p{N}])(l|d|qu)['\x{2019}]", r"\1\2 ")  # French elisions
    x = pc.replace_substring_regex(x, r"['\x{2019}\x{2018}`\x{00B4}]", "")
    if mode == "addr":
        x = pc.replace_substring_regex(x, r"<null>|(^|[^\p{L}])n/a([^\p{L}]|$)", r"\1 \2")
        x = pc.replace_substring(x, "#", "")
        x = pc.replace_substring_regex(x, r"[^\p{L}\p{M}\p{N}\s,]+", " ")
        x = pc.replace_substring_regex(x, r"\s*,\s*", ",")
    else:
        x = pc.replace_substring_regex(x, r"[^\p{L}\p{M}\p{N}\s]+", " ")
    x = pc.replace_substring_regex(x, r"\s+", " ")
    return pc.utf8_trim_whitespace(x)


# ----------------------------------------------------------------- names
_VOWELS_AFTER_FIRST = re.compile(r"(?<=.)[aeiou]")
_H_AFTER_CONS = re.compile(r"(?<=[bcdfgjklmnpqrstvxyz])h")
_REPEAT = re.compile(r"(.)\1+")
_WEB_RAW = re.compile(r"www\.|https?://|@|\.(?:com|net|org|in|co|fr|io|biz|info|us)\b", re.I)
_NAME_DROP = R.HONORIFIC_TOKENS | R.WEB_TOKENS | R.LEGAL_TOKENS | R.NAME_STOPWORDS
_MARKER_FIRST = {mk[0] for mk in R.NAME_MARKERS}


@lru_cache(maxsize=1_000_000)
def phon(tok: str) -> str:
    """Crude phonetic key of one token (see module docstring of the task spec)."""
    t = tok
    for a, b in (("aa", "a"), ("ee", "i"), ("ii", "i"), ("oo", "u"), ("uu", "u"), ("ph", "f"), ("ck", "k")):
        t = t.replace(a, b)
    t = t.replace("w", "v").replace("z", "j").replace("q", "k").replace("x", "ks")
    t = _H_AFTER_CONS.sub("", t)
    t = _VOWELS_AFTER_FIRST.sub("", t)
    t = _REPEAT.sub(r"\1", t)
    return t or tok


_LEET_TABLE = str.maketrans(R.LEET_DIGITS)
_HAS_ALPHA = re.compile(r"[a-z]")
_HAS_DIGIT = re.compile(r"\d")


def _unleet(t: str) -> str:
    """'5ervices' -> 'services', 'c0m' -> 'com' (tokens of >= 3 chars mixing letters and
    digits, except ordinals / unit-like tokens such as '3rd', 'a1')."""
    if len(t) >= 3 and _HAS_DIGIT.search(t) and _HAS_ALPHA.search(t) and not R.ORDINAL_RE.match(t):
        return t.translate(_LEET_TABLE)
    return t


def _drop_marker(tokens: list[str]) -> tuple[list[str], list[str], bool]:
    """Split tokens at the first trade/former-name marker.

    Returns (tokens without the marker phrase, tokens after the marker, found?).
    """
    if _MARKER_FIRST.isdisjoint(tokens):  # fast path: no marker can start here
        return tokens, [], False
    n = len(tokens)
    for i in range(n):
        for mk in R.NAME_MARKERS:
            k = len(mk)
            if tuple(tokens[i:i + k]) == mk:
                return tokens[:i] + tokens[i + k:], tokens[i + k:], True
    return tokens, [], False


def _core_tokens(tokens: list[str]) -> list[str]:
    """Remove honorifics (incl. the 'm s' bigram), web parts, legal forms, stop
    words, long digit runs and immediately repeated tokens."""
    out: list[str] = []
    i, n = 0, len(tokens)
    while i < n:
        t = tokens[i]
        if t == "m" and i + 1 < n and tokens[i + 1] == "s":
            i += 2
            continue
        i += 1
        if t in _NAME_DROP or (len(t) >= 7 and t.isdigit()):
            continue
        if out and out[-1] == t:
            continue
        out.append(t)
    return out


class NameProcessor:
    """Builds name features; ``vocab`` enables glued-word segmentation.

    Args:
        vocab: S1 name_clean token counts of the same split+country (None for S1).
    """

    def __init__(self, vocab: Counter | None = None):
        self.logp = None
        self._seg_cache: dict[str, list[str] | None] = {}
        if vocab:
            total = sum(vocab.values())
            self.logp = {w: math.log(c / total) for w, c in vocab.items()
                         if len(w) >= R.SEG_MIN_PIECE and c >= R.SEG_MIN_COUNT}
            self.maxlen = max(len(w) for w in self.logp)

    def segment(self, tok: str) -> list[str] | None:
        """Best split of ``tok`` into >= 2 vocabulary words (pieces >= 2 chars),
        maximising the summed log-frequency; None if no full segmentation exists or
        the pieces are too short on average (junk tokens like "tavosoljax")."""
        if tok in self._seg_cache:
            return self._seg_cache[tok]
        n = len(tok)
        best = [-math.inf] * (n + 1)
        back = [0] * (n + 1)
        best[0] = 0.0
        for i in range(2, n + 1):
            for j in range(max(0, i - self.maxlen), i - 1):
                if best[j] == -math.inf:
                    continue
                w = self.logp.get(tok[j:i])
                if w is not None and best[j] + w > best[i]:
                    best[i], back[i] = best[j] + w, j
        res = None
        if best[n] > -math.inf:
            pieces, i = [], n
            while i > 0:
                pieces.append(tok[back[i]:i])
                i = back[i]
            if len(pieces) >= 2 and n / len(pieces) >= R.SEG_MIN_AVG_PIECE:
                res = pieces[::-1]
        self._seg_cache[tok] = res
        return res

    def _glue(self, tokens: list[str]) -> list[str]:
        if self.logp is None:
            return tokens
        out = []
        for t in tokens:
            if len(t) >= 7 and t not in self.logp and not t.isdigit():
                seg = self.segment(t)
                if seg:
                    out.extend(seg)
                    continue
            out.append(t)
        return out

    def features(self, clean_for_core: str) -> tuple[str, str, str, str, bool]:
        """(name_core, name_alt, name_concat, name_phon, has_marker) from the cleaned
        name (phone numbers and #tags already removed)."""
        tokens = [_unleet(t) for t in clean_for_core.split()]
        main, after, marker = _drop_marker(tokens)
        core = _core_tokens(self._glue(main))
        if not core:  # e.g. "The Company Ltd": keep something to match on
            core = [t for t in main if t not in R.HONORIFIC_TOKENS] or main
        alt = _core_tokens(self._glue(after)) if marker else []
        return (" ".join(core), " ".join(alt), "".join(core), " ".join(phon(t) for t in core), marker)


# ----------------------------------------------------------------- addresses
_DIGITS = re.compile(r"\d+")


def address_features(s: str, country: str) -> tuple:
    """(addr_clean, addr_core, state, addr_nums, house_num, empty, landmark, pobox)."""
    comps = []
    for c in s.split(","):
        toks = [t for t in c.split() if t != "null"]
        c = " ".join(toks)
        if c and c not in R.ADDR_PLACEHOLDERS:
            comps.append(c)
    addr_clean = " ".join(comps)
    lookup = R.for_country(R.STATE_LOOKUP, country)
    state = ""
    for i in range(len(comps) - 1, -1, -1):
        code = lookup.get(comps[i])
        if code:
            state = code
            del comps[i]
            break
    raw_tokens = " ".join(comps).split()
    abbr = R.for_country(R.ABBREVIATIONS, country)
    bigr = R.for_country(R.ABBREVIATION_BIGRAMS, country)
    tokens: list[str] = []
    i, n = 0, len(raw_tokens)
    while i < n:
        t = raw_tokens[i]
        if i + 1 < n and (t, raw_tokens[i + 1]) in bigr:
            tokens.extend(bigr[(t, raw_tokens[i + 1])].split())
            i += 2
            continue
        tokens.extend(abbr[t].split() if t in abbr else (t,))
        i += 1
    addr_core = " ".join(tokens)
    nums = [str(int(d)) for d in _DIGITS.findall(addr_core)]
    tset = set(raw_tokens) | set(tokens)
    landmark = bool(tset & R.LANDMARK_TOKENS) or any(_has_seq(tokens, p) for p in R.LANDMARK_PHRASES)
    pobox = any(_has_seq(raw_tokens, p) for p in R.POBOX_PATTERNS)
    return (addr_clean, addr_core, state, " ".join(nums), nums[0] if nums else None,
            not addr_clean, landmark, pobox)


def _has_seq(tokens: list[str], seq: tuple) -> bool:
    """True if ``seq`` occurs contiguously in ``tokens``."""
    k = len(seq)
    if k == 1:
        return seq[0] in tokens
    return any(tuple(tokens[i:i + k]) == seq for i in range(len(tokens) - k + 1))


# ----------------------------------------------------------------- driver
SCHEMA = pa.schema([
    ("src", pa.int8()), ("idx", pa.int32()), ("entity_id", LARGE), ("country", LARGE),
    ("name_raw", LARGE), ("addr_raw", LARGE),
    ("name_clean", LARGE), ("name_core", LARGE), ("name_alt", LARGE), ("name_concat", LARGE),
    ("name_phon", LARGE), ("name_has_indic", pa.bool_()), ("name_has_marker", pa.bool_()),
    ("name_is_web", pa.bool_()),
    ("addr_clean", LARGE), ("addr_core", LARGE), ("state", LARGE), ("addr_nums", LARGE),
    ("house_num", LARGE), ("addr_empty", pa.bool_()), ("addr_has_landmark", pa.bool_()),
    ("addr_has_pobox", pa.bool_()),
])


def process_source(src: int, cols: dict, tr: Transliterator, vocabs: dict[str, Counter] | None) -> pa.Table:
    """Normalise one source's records; returns a table with :data:`SCHEMA`."""
    n = len(cols["id"])
    log(f"  source {src}: cleaning {n:,} records")
    name_clean = clean_text(cols["name"], "name", tr)
    name_pre = clean_text(pc.replace_substring_regex(cols["name"], _PHONE_TAG_RE2, " "), "name", tr)
    addr_pre = clean_text(cols["addr"], "addr", tr)
    has_indic = pc.fill_null(pc.match_substring_regex(cols["name"], r"[\x{0900}-\x{0D7F}]"), False)
    is_web = pc.fill_null(pc.match_substring_regex(
        cols["name"], r"(?i)www\.|https?://|@|\.(com|net|org|in|co|fr|io|biz|info|us)\b"), False)
    country = cols["country"].to_pylist()
    procs = {c: NameProcessor(vocabs.get(c) if vocabs else None) for c in set(country)}
    log(f"  source {src}: token features")
    out = {k: [] for k in ("name_core", "name_alt", "name_concat", "name_phon", "name_has_marker",
                           "addr_clean", "addr_core", "state", "addr_nums", "house_num",
                           "addr_empty", "addr_has_landmark", "addr_has_pobox")}
    for s in range(0, n, CHUNK):
        names = name_pre.slice(s, CHUNK).to_pylist()
        addrs = addr_pre.slice(s, CHUNK).to_pylist()
        for nm, ad, c in zip(names, addrs, country[s:s + CHUNK]):
            core, alt, concat, ph, mk = procs[c].features(nm)
            out["name_core"].append(core)
            out["name_alt"].append(alt)
            out["name_concat"].append(concat)
            out["name_phon"].append(ph)
            out["name_has_marker"].append(mk)
            a = address_features(ad, c)
            for k, v in zip(("addr_clean", "addr_core", "state", "addr_nums", "house_num",
                             "addr_empty", "addr_has_landmark", "addr_has_pobox"), a):
                out[k].append(v)
    tbl = {
        "src": pa.array(np.full(n, src, np.int8)), "idx": pa.array(np.arange(n, dtype=np.int32)),
        "entity_id": cols["id"], "country": cols["country"], "name_raw": cols["name"], "addr_raw": cols["addr"],
        "name_clean": name_clean, "name_has_indic": has_indic, "name_is_web": is_web,
    }
    for k, v in out.items():
        tbl[k] = pa.array(v, SCHEMA.field(k).type)
    return pa.table({f.name: tbl[f.name] for f in SCHEMA}, schema=SCHEMA)


def s1_vocab(name_clean: pa.Array, country: pa.Array) -> dict[str, Counter]:
    """S1 name_clean token counts per country (the segmentation vocabulary)."""
    vocabs: dict[str, Counter] = {}
    for nm, c in zip(name_clean.to_pylist(), country.to_pylist()):
        vocabs.setdefault(c, Counter()).update(nm.split())
    return vocabs


def normalize_split(split: str) -> None:
    """Normalise all three sources of a split into cache/{split}_records.parquet."""
    CACHE.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tr = Transliterator.load()
    log(f"{split}: translit dict {len(tr.token_dict):,} tokens, state map {len(tr.state_map):,}")
    sources = load_sources(split)
    path = CACHE / f"{split}_records.parquet"
    with pq.ParquetWriter(path, SCHEMA, compression="zstd") as w:
        t1 = process_source(1, sources[1], tr, None)
        vocabs = s1_vocab(t1.column("name_clean").combine_chunks(), t1.column("country").combine_chunks())
        w.write_table(t1, row_group_size=500_000)
        del t1
        for src in (2, 3):
            t = process_source(src, sources[src], tr, vocabs)
            w.write_table(t, row_group_size=500_000)
            del t
            sources[src] = None
    log(f"{split}: wrote {path} (peak RSS {peak_rss():.1f} GiB)")
    (CACHE / f"{split}_normalize_stats.json").write_text(json.dumps(
        {"seconds": time.time() - t0, "peak_rss_gib": peak_rss()}), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Normalise records -> cache/{split}_records.parquet")
    ap.add_argument("--split", choices=["train", "test"], required=True)
    normalize_split(ap.parse_args().split)


if __name__ == "__main__":
    main()
