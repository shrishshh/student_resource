"""All hand-written rule lists used by the pipeline (no external data).

Every table that depends on the country is a dict keyed by the country label as
it appears in the data. Unknown labels fall back to ``GENERIC`` (empty tables:
no state detection, no abbreviation expansion) via :func:`for_country`, so the
code never branches on a hard-coded country list.
"""

from __future__ import annotations

import re

GENERIC = "__generic__"


def for_country(table: dict, country: str):
    """Country-specific entry of a rule table, or the generic fallback."""
    return table.get(country, table[GENERIC])


# --------------------------------------------------------------------------- names
# Honorific / prefix tokens dropped from name_core. ("m s" comes from "M/S" = Messrs.)
HONORIFIC_TOKENS = {"ms", "mr", "mrs", "smt", "shri", "sri", "shree", "sree", "dr"}  # "sree": learned romanisation of श्री
HONORIFIC_BIGRAMS = {("m", "s")}

# Legal forms (applied to every country: noisy sources move/translate them freely).
LEGAL_US = {"inc", "incorporated", "llc", "lp", "llp", "pllc", "pc", "plc", "ltd", "limited", "corp",
            "corporation", "co", "company", "ag", "aktiengesellschaft", "gmbh"}
LEGAL_IN = {"pvt", "private", "ltd", "limited", "llp", "opc"}
LEGAL_FR = {"sa", "sas", "sasu", "sarl", "eurl", "ei", "sci", "snc", "scop", "selarl", "sca", "gie"}
# Rule-based romanisations of Indic legal words that the learned dictionary may miss
# (outputs of translit.romanize for the private/limited/LLP words in the 9 scripts).
LEGAL_TRANSLIT = {"praivet", "praibhet", "piraivet", "praivat", "praivett", "praivettu", "pra", "li",
                  "limitet", "limited", "limitad", "limittad", "limtid", "limitedu", "limited",
                  "elelpi", "elelapi", "elpi", "pvt"}
LEGAL_TOKENS = LEGAL_US | LEGAL_IN | LEGAL_FR | LEGAL_TRANSLIT

# Function words dropped from name_core.
NAME_STOPWORDS = {"and", "the", "of", "de", "du", "des", "la", "le", "les", "et", "l", "d", "qu"}

# Web parts dropped from name_core (tokens after punctuation removal).
WEB_TOKENS = {"www", "com", "net", "org", "in", "co", "fr", "io", "biz", "info", "us", "http", "https"}

# Trade / former-name markers (cleaned token sequences). The real name follows the marker.
# Longest first so "formerly known as" wins over "formerly".
NAME_MARKERS = sorted([
    ("formerly", "known", "as"), ("formerly",), ("fka",), ("f", "k", "a"), ("nee",),
    ("trading", "as"), ("t", "a"), ("doing", "business", "as"), ("dba",), ("d", "b", "a"),
    ("aka",), ("a", "k", "a"),
], key=len, reverse=True)

# Glued-word segmentation (name tokens >= 7 chars missing from the S1 vocabulary):
# pieces must be S1 vocabulary words of >= SEG_MIN_PIECE chars seen >= SEG_MIN_COUNT times,
# with an average piece length >= SEG_MIN_AVG_PIECE (rejects junk like "ta vo sol jax").
SEG_MIN_PIECE, SEG_MIN_COUNT, SEG_MIN_AVG_PIECE = 2, 3, 4.0

# Leet-style digit substitutions seen in noisy names ("5ervices", "c0m"); applied only to
# tokens mixing letters and digits that are not ordinals / unit-like (3rd, 12b, a1).
LEET_DIGITS = {"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b"}
ORDINAL_RE = re.compile(r"^\d+(st|nd|rd|th|[a-z])?$|^[a-z]\d+$")

# --------------------------------------------------------------------------- addresses
# Placeholder components/tokens removed from addresses.
ADDR_PLACEHOLDERS = {"null", "n a"}  # whole components ("<NULL>", "n/a", "null") and "null" tokens

LANDMARK_TOKENS = {"near", "nr", "opp", "opposite", "behind", "beside", "adjacent", "pres", "face", "cote"}
LANDMARK_PHRASES = [("next", "to"), ("in", "front", "of")]
POBOX_PATTERNS = [("po", "box"), ("pmb",), ("bp",), ("p", "o", "box"), ("post", "box")]

# Country-keyed abbreviation expansion (single tokens -> one or more tokens).
ABBREVIATIONS = {
    GENERIC: {},
    "US": {"st": "street", "rd": "road", "dr": "drive", "ave": "avenue", "av": "avenue", "ln": "lane",
           "ct": "court", "blvd": "boulevard", "pl": "place", "pkwy": "parkway", "hwy": "highway",
           "cir": "circle", "ste": "unit", "apt": "unit", "fl": "floor", "n": "north", "s": "south",
           "e": "east", "w": "west"},
    "India": {"rd": "road", "ngr": "nagar", "flt": "flat", "hno": "house no", "opp": "opposite",
              "nr": "near", "bldg": "building", "sec": "sector", "extn": "extension"},
    "France": {"r": "rue", "av": "avenue", "bd": "boulevard", "ch": "chemin", "pl": "place", "imp": "impasse",
               "all": "allee", "rte": "route", "fg": "faubourg", "fbg": "faubourg", "sq": "square",
               "qu": "quai", "crs": "cours", "pass": "passage", "res": "residence", "bat": "batiment",
               "st": "saint", "ste": "sainte"},
}
# Two-token abbreviations (checked before single tokens).
ABBREVIATION_BIGRAMS = {
    GENERIC: {},
    "US": {},
    "India": {("h", "no"): "house no"},
    "France": {},
}

# --------------------------------------------------------------------------- states / regions
# canonical code -> surface forms (normalised: lower-case, no accents, punctuation -> space).
_US_STATES = {
    "al": ["alabama"], "ak": ["alaska"], "az": ["arizona"], "ar": ["arkansas"], "ca": ["california"],
    "co": ["colorado"], "ct": ["connecticut"], "de": ["delaware"], "dc": ["district of columbia", "d c"],
    "fl": ["florida"], "ga": ["georgia"], "hi": ["hawaii"], "id": ["idaho"], "il": ["illinois"],
    "in": ["indiana"], "ia": ["iowa"], "ks": ["kansas"], "ky": ["kentucky"], "la": ["louisiana"],
    "me": ["maine"], "md": ["maryland"], "ma": ["massachusetts"], "mi": ["michigan"], "mn": ["minnesota"],
    "ms": ["mississippi"], "mo": ["missouri"], "mt": ["montana"], "ne": ["nebraska"], "nv": ["nevada"],
    "nh": ["new hampshire"], "nj": ["new jersey"], "nm": ["new mexico"], "ny": ["new york"],
    "nc": ["north carolina"], "nd": ["north dakota"], "oh": ["ohio"], "ok": ["oklahoma"], "or": ["oregon"],
    "pa": ["pennsylvania"], "ri": ["rhode island"], "sc": ["south carolina"], "sd": ["south dakota"],
    "tn": ["tennessee"], "tx": ["texas"], "ut": ["utah"], "vt": ["vermont"], "va": ["virginia"],
    "wa": ["washington"], "wv": ["west virginia"], "wi": ["wisconsin"], "wy": ["wyoming"],
    "pr": ["puerto rico"], "gu": ["guam"], "vi": ["virgin islands"],
}
_IN_STATES = {
    "an": ["andaman and nicobar islands", "andaman and nicobar"], "ap": ["andhra pradesh"],
    "ar": ["arunachal pradesh"], "as": ["assam"], "br": ["bihar"], "ch": ["chandigarh"],
    "cg": ["chhattisgarh", "chattisgarh", "ct"], "dn": ["dadra and nagar haveli", "dadra and nagar haveli and daman and diu"],
    "dd": ["daman and diu"], "dl": ["delhi", "nct of delhi"], "ga": ["goa"], "gj": ["gujarat"],
    "hr": ["haryana"], "hp": ["himachal pradesh"], "jk": ["jammu and kashmir", "jammu kashmir"],
    "jh": ["jharkhand"], "ka": ["karnataka"], "kl": ["kerala", "keralam"], "la": ["ladakh"],
    "ld": ["lakshadweep"], "mp": ["madhya pradesh"], "mh": ["maharashtra"], "mn": ["manipur"],
    "ml": ["meghalaya"], "mz": ["mizoram"], "nl": ["nagaland"], "or": ["odisha", "orissa", "od"],
    "py": ["puducherry", "pondicherry"], "pb": ["punjab"], "rj": ["rajasthan"], "sk": ["sikkim"],
    "tn": ["tamil nadu", "tamilnadu"], "tg": ["telangana", "ts"], "tr": ["tripura"],
    "up": ["uttar pradesh"], "uk": ["uttarakhand", "uttaranchal", "ut"], "wb": ["west bengal"],
}
# France: regions (canonical) + departments seen as trailing components in the test
# files (Nord, Pas-de-Calais, Gironde, Loire-Atlantique) and the other departments of
# the same regions, each mapped to its region.
_FR_REGIONS = {
    "hdf": ["hauts de france", "nord", "pas de calais", "aisne", "oise", "somme", "nord pas de calais", "picardie"],
    "naq": ["nouvelle aquitaine", "gironde", "charente", "charente maritime", "correze", "creuse", "dordogne",
            "landes", "lot et garonne", "pyrenees atlantiques", "deux sevres", "vienne", "haute vienne", "aquitaine"],
    "pdl": ["pays de la loire", "loire atlantique", "maine et loire", "mayenne", "sarthe", "vendee"],
    "idf": ["ile de france", "paris", "seine et marne", "yvelines", "essonne", "hauts de seine",
            "seine saint denis", "val de marne", "val d oise"],
    "bre": ["bretagne"], "nor": ["normandie"], "ges": ["grand est"], "bfc": ["bourgogne franche comte"],
    "cvl": ["centre val de loire"], "ara": ["auvergne rhone alpes"], "occ": ["occitanie"],
    "pac": ["provence alpes cote d azur", "provence alpes cote dazur"], "cor": ["corse"],
}


def _lookup(table: dict[str, list[str]], include_codes: bool) -> dict[str, str]:
    """surface form -> canonical code (codes themselves included when requested)."""
    out = {}
    for code, names in table.items():
        if include_codes:
            out[code] = code
        for n in names:
            out[n] = code
    return out


# surface form (a whole address component, normalised) -> canonical state code.
STATE_LOOKUP = {
    GENERIC: {},
    "US": _lookup(_US_STATES, True),
    "India": _lookup(_IN_STATES, True),
    "France": _lookup(_FR_REGIONS, False),  # 2-3 letter codes are not used in French addresses
}
# canonical code -> readable canonical name (used when a native-script state is mapped).
STATE_NAME = {
    "US": {c: n[0] for c, n in _US_STATES.items()},
    "India": {c: n[0] for c, n in _IN_STATES.items()},
    "France": {c: n[0] for c, n in _FR_REGIONS.items()},
}
