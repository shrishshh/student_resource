"""Vectorised (pyarrow) text normalisation used by the EDA.

All functions take and return ``pyarrow`` string arrays so they run in C++ over
millions of rows. Regexes are RE2 syntax (pyarrow.compute), not Python ``re``.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc

# Legal-form / filler tokens removed to build the "stripped name".
# ("m s" comes from "M/S" = Messrs after punctuation -> space.)
LEGAL_TOKENS = [
    "inc", "incorporated", "corp", "corporation", "co", "company", "llc", "ltd",
    "limited", "pvt", "private", "plc", "llp", "the", "m", "s",
]
_LEGAL_RE = r" (?:" + "|".join(LEGAL_TOKENS) + r") "

# Landmark words on the noisy side (matched as whole tokens of the normalised address).
LANDMARK_RE = (
    r"(?:^| )(?:near|nr|opp|opposite|behind|beside|next to|adjacent|in front of|above)(?: |$)"
)
# Trade-name markers, matched on the lower-cased RAW name (before punctuation removal).
DBA_RE = (
    r"\bdba\b|\bd/b/a\b|\bd\.b\.a\b|\bt/a\b|\btrading as\b|\bdoing business as\b|\bm/s\b"
)


def normalize(arr: pa.Array) -> pa.Array:
    """Lower-case, NFKD accent strip, punctuation/symbols -> space, collapse spaces.

    Only combining diacritics in U+0300-U+036F are removed, so accented Latin letters
    lose their accents while Indic scripts (e.g. Devanagari vowel signs) stay intact.
    """
    x = pc.utf8_lower(arr)
    x = pc.utf8_normalize(x, "NFKD")
    x = pc.replace_substring_regex(x, r"[\x{0300}-\x{036f}]", "")
    x = pc.replace_substring_regex(x, r"[\p{P}\p{S}\s]+", " ")
    return pc.utf8_trim_whitespace(x)


def strip_legal(norm: pa.Array) -> pa.Array:
    """Remove :data:`LEGAL_TOKENS` (whole tokens) from an already normalised name."""
    sp = pa.scalar(" ", pa.large_string())
    x = pc.binary_join_element_wise(sp, norm, sp, pa.scalar("", pa.large_string()))
    while True:  # repeat: adjacent legal tokens ("pvt ltd") share a space
        y = pc.replace_substring_regex(x, _LEGAL_RE, " ")
        if pc.all(pc.equal(x, y)).as_py():
            break
        x = y
    return pc.utf8_trim_whitespace(x)


def postal_like(norm_addr: pa.Array) -> pa.Array:
    """Last standalone 5-6 digit run of a normalised address (null if none).

    "ddd ddd" (e.g. Indian PIN "110 001") is first joined into one 6-digit run.
    """
    x = pc.replace_substring_regex(norm_addr, r"(^|\D)(\d{3}) (\d{3})(\D|$)", r"\1\2\3\4")
    m = pc.extract_regex(x, r"^.*(?:^|\D)(?P<p>\d{5,6})(?:\D|$)")
    return pc.struct_field(m, [0])


def house_number(norm_addr: pa.Array) -> pa.Array:
    """First standalone 1-4 digit run of a normalised address (null if none)."""
    m = pc.extract_regex(norm_addr, r"(?:^|\D)(?P<h>\d{1,4})(?:\D|$)")
    return pc.struct_field(m, [0])
