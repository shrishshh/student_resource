"""Exploratory data analysis -> reports/eda_report.md.

Run from code/business_entity_resolution/:
    python -m src.eda                        # full data
    python -m src.eda --limit 200000   # quick debug run

Sections (letters match the report): A machine, B inventory, C provided docs,
D integrity, E country labels, F sizes, G ground truth, H true-pair diagnostics,
I token substitutions, J hard negatives, K raw samples, plus the all-empty
submission format check. Text normalisation lives in ``text_norm.py``.

Design notes
------------
* ~24M rows on a 16 GB laptop, so every column is held once, as a pyarrow array
  (Source 2 + Source 3 are a zero-copy ChunkedArray). Lookups use Arrow hash
  kernels (``index_in``, ``value_counts``, ``is_in``) instead of Python dicts, and
  regex work is vectorised (RE2 via pyarrow.compute). Python loops are only used
  for token-set logic, in chunks.
* Train and test are processed one after the other.
* Every random choice uses ``numpy.random.default_rng(42)``.
* Nothing under dataset/ or utils/ is modified.
"""

from __future__ import annotations

import argparse
import gc
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.compute as pc
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
from scipy.stats import spearmanr

from . import text_norm as tn
from .io_utils import DATASET_DIR, PROJECT_ROOT, load_ground_truth, read_tsv
from .make_empty_submission import write_all_empty
from .metrics import macro_f05

SEED = 42
REPORT_PATH = PROJECT_ROOT / "reports" / "eda_report.md"
EMPTY_SUB_DIR = PROJECT_ROOT / "submissions" / "00_all_empty"
SKIP_DIRS = {"__MACOSX", ".venv", ".git", "__pycache__", ".pytest_cache"}
NEG_SAMPLE_TARGET = 300_000  # max sampled negative pairs used for quantiles/examples
CHUNK = 1_000_000
COLS = {"entity_id": "id", "business_name": "name", "business_address": "addr", "country": "country"}
LARGE = pa.large_string()

T0 = time.time()


# --------------------------------------------------------------------------- utils
def log(msg: str) -> None:
    """Progress line on stderr with elapsed seconds and process RSS."""
    rss = psutil.Process().memory_info().rss / 2**30
    print(f"[{time.time() - T0:7.1f}s | {rss:4.1f} GiB] {msg}", file=sys.stderr, flush=True)


def pct(n: float, d: float) -> str:
    """``"n (p%)"`` with thousands separators."""
    return f"{int(n):,} ({(100.0 * n / d) if d else 0.0:.2f}%)"


def p2(n: float, d: float) -> str:
    """Percentage only."""
    return f"{(100.0 * n / d) if d else 0.0:.2f}%"


def cell(v) -> str:
    """Escape a value for a markdown table cell (full text, never truncated)."""
    s = str(v)
    if s == "":
        return "*(empty)*"
    # make invisible control characters (e.g. mojibake U+0080) visible as \xNN
    s = re.sub(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]", lambda m: f"\\x{ord(m.group()):02x}", s)
    return s.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def md_table(headers: list, rows: list[list]) -> str:
    """Render a markdown table."""
    out = ["| " + " | ".join(cell(h) for h in headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(cell(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def human(nbytes: float) -> str:
    """Human-readable byte size."""
    for unit in ("B", "KB", "MB"):
        if nbytes < 1024:
            return f"{nbytes:.0f} B" if unit == "B" else f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.2f} GB"


def arrow(series: pd.Series) -> pa.Array:
    """pandas string Series -> one contiguous pyarrow large_string Array."""
    a = pa.array(series)
    if isinstance(a, pa.ChunkedArray):
        a = a.combine_chunks()
    return a.cast(LARGE)


def to_np_bool(arr) -> np.ndarray:
    """pyarrow boolean (nulls -> False) to numpy bool."""
    return np.asarray(pc.fill_null(arr, False).to_numpy(zero_copy_only=False), dtype=bool)


def nonempty(arr) -> np.ndarray:
    """numpy mask of non-empty strings."""
    return to_np_bool(pc.not_equal(arr, ""))


def regex_mask(arr, pattern: str) -> np.ndarray:
    """numpy mask of strings matching an RE2 pattern."""
    return to_np_bool(pc.match_substring_regex(arr, pattern))


def filt(arr, mask: np.ndarray):
    """Filter an Array/ChunkedArray with a numpy bool mask."""
    return pc.filter(arr, pa.array(mask))


def take(arr, idx: np.ndarray):
    """Gather positions ``idx`` (numpy ints) from an Array/ChunkedArray."""
    return arr.take(pa.array(idx, type=pa.int64()))


def lookup(values, value_set) -> np.ndarray:
    """Position of each value inside ``value_set`` (-1 if absent), via Arrow hashing."""
    if isinstance(value_set, pa.ChunkedArray):
        value_set = value_set.combine_chunks()
    return pc.fill_null(pc.index_in(values, value_set=value_set), -1).to_numpy(zero_copy_only=False)


def cat_np(arr) -> np.ndarray:
    """Low-memory numpy object array of a low-cardinality string column.

    Elements share one Python str per distinct value (per chunk), so the cost is
    8 bytes/row instead of one str object per row.
    """
    chunks = arr.chunks if isinstance(arr, pa.ChunkedArray) else [arr]
    parts = []
    for ch in chunks:
        enc = pc.dictionary_encode(ch)
        dic = np.array(enc.dictionary.to_pylist() + [None], dtype=object)
        parts.append(dic[pc.fill_null(enc.indices, len(dic) - 1).to_numpy(zero_copy_only=False)])
    return np.concatenate(parts) if parts else np.array([], dtype=object)


def value_counts(arr) -> list[tuple[str, int]]:
    """Exact value counts, most frequent first."""
    vc = pc.value_counts(arr)
    pairs = list(zip(vc.field("values").to_pylist(), vc.field("counts").to_pylist()))
    return sorted(pairs, key=lambda t: (-t[1], str(t[0])))


def tsr(a, b) -> np.ndarray:
    """Element-wise rapidfuzz token_set_ratio of two aligned string arrays."""
    out = np.empty(len(a), dtype=np.float32)
    for s in range(0, len(a), CHUNK):
        out[s:s + CHUNK] = cpdist(a.slice(s, CHUNK).to_pylist(), b.slice(s, CHUNK).to_pylist(),
                                  scorer=fuzz.token_set_ratio, workers=-1)
    return out


def quantiles(x: np.ndarray, qs) -> list[str]:
    """Formatted percentiles (or '-' when empty)."""
    if len(x) == 0:
        return ["-"] * len(qs)
    return [f"{v:.1f}" for v in np.percentile(x, qs)]


def script_of(ch: str) -> str:
    """Script-ish label of a non-ASCII char: first word of its Unicode name (letters/marks)."""
    if unicodedata.category(ch)[0] not in "LM":
        return "non-letter symbol/punct"
    word = unicodedata.name(ch, "UNKNOWN").split(" ")[0]
    return "LATIN" if word in ("MASCULINE", "FEMININE") else word  # º ª ordinal indicators


def rec(d: dict, i: int) -> list:
    """[id, name, address, country] of record ``i``."""
    return [d[k][int(i)].as_py() for k in ("id", "name", "addr", "country")]


# --------------------------------------------------------------------------- data
class Split:
    """One data split held as pyarrow arrays (each column stored once).

    Attributes:
        sources: {"S1"|"S2"|"S3": {"id","name","addr","country"}} raw arrays.
        s1: Source-1 arrays plus normalised columns ``nname``, ``sname``, ``naddr``
            and ``country_np`` (numpy object).
        r:  Source 2 + Source 3 as zero-copy ChunkedArrays (S2 rows first), same
            columns plus numpy ``src`` ('S2'/'S3') and ``row`` (row in its own file).
        countries: sorted country labels present in any source.
    """

    def __init__(self, name: str, limit: int | None):
        self.name = name
        self.sources: dict[str, dict] = {}
        for i in (1, 2, 3):
            log(f"loading {name}_source{i}")
            df = read_tsv(DATASET_DIR / name / f"{name}_source{i}.tsv")
            if limit:
                df = df.iloc[:limit]
            self.sources[f"S{i}"] = {short: arrow(df[col]) for col, short in COLS.items()}
            del df
            gc.collect()
        self.s1 = dict(self.sources["S1"])
        s2, s3 = self.sources["S2"], self.sources["S3"]
        self.r = {k: pa.chunked_array([s2[k], s3[k]]) for k in s2}
        n2, n3 = len(s2["id"]), len(s3["id"])
        self.r["src"] = np.empty(n2 + n3, dtype=object)
        self.r["src"][:n2], self.r["src"][n2:] = "S2", "S3"
        self.r["row"] = np.concatenate([np.arange(n2), np.arange(n3)])
        log(f"normalising {name}")
        for d in (self.s1, self.r):
            d["nname"] = tn.normalize(d["name"])
            d["sname"] = tn.strip_legal(d["nname"])
            d["naddr"] = tn.normalize(d["addr"])
            d["country_np"] = cat_np(d["country"])
        self.countries = sorted(set(self.s1["country_np"]) | set(self.r["country_np"]))
        log(f"{name}: S1={len(self.s1['id']):,} S2={n2:,} S3={n3:,}")


# ------------------------------------------------------------------ section A / B / C
def section_machine() -> str:
    """A. Machine description."""
    cpu = platform.processor()
    if sys.platform == "win32":
        try:
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                               r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            cpu = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    elif Path("/proc/cpuinfo").exists():
        m = re.search(r"model name\s*:\s*(.*)", Path("/proc/cpuinfo").read_text())
        cpu = m.group(1) if m else cpu
    gpu = "none (nvidia-smi not found)"
    if shutil.which("nvidia-smi"):
        try:
            q = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                                "--format=csv,noheader"], capture_output=True, text=True).stdout
            full = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
            cuda = re.search(r"CUDA Version:\s*([\d.]+)", full)
            gpu = f"{q.strip()}; CUDA {cuda.group(1) if cuda else '?'}"
        except OSError as exc:
            gpu = f"nvidia-smi failed: {exc}"
    vm = psutil.virtual_memory()
    du = shutil.disk_usage(PROJECT_ROOT)
    rows = [
        ["OS", f"{platform.system()} {platform.release()} (build {platform.version()})"],
        ["CPU", cpu],
        ["Cores", f"{psutil.cpu_count(logical=False)} physical / {psutil.cpu_count()} logical"],
        ["RAM", f"{vm.total / 2**30:.1f} GiB total, {vm.available / 2**30:.1f} GiB available at start"],
        ["GPU", gpu],
        ["Python", sys.version.replace("\n", " ")],
        ["Key packages", f"pandas {pd.__version__}, numpy {np.__version__}, pyarrow {pa.__version__}"],
        ["Free disk", f"{du.free / 2**30:.1f} GiB free of {du.total / 2**30:.1f} GiB ({PROJECT_ROOT.anchor})"],
    ]
    return "## A. Machine\n\n" + md_table(["Item", "Value"], rows)


def raw_scan(path: Path) -> dict:
    """Byte-level scan of a TSV: line counts, CR/BOM, tab-count anomalies, first lines."""
    n_lines = blank = cr = quote_lines = 0
    bad_tabs: Counter = Counter()
    first: list[bytes] = []
    header_tabs = None
    last = b""
    with open(path, "rb") as f:
        for line in f:
            n_lines += 1
            last = line
            if len(first) < 3:
                first.append(line)
            if b"\r" in line:
                cr += 1
            if b'"' in line:
                quote_lines += 1
            if not line.strip(b"\r\n"):
                blank += 1
                continue
            t = line.count(b"\t")
            if header_tabs is None:
                header_tabs = t
            elif t != header_tabs:
                bad_tabs[t] += 1
    return {"lines": n_lines, "blank": blank, "cr": cr, "quote_lines": quote_lines,
            "bad_tabs": dict(bad_tabs), "first": first,
            "bom": bool(first) and first[0].startswith(b"\xef\xbb\xbf"),
            "trailing_newline": last.endswith(b"\n"),
            "columns": first[0].decode("utf-8").rstrip("\r\n").split("\t") if first else []}


def section_inventory(parsed_rows: dict[str, int | None], scans: dict[str, dict]) -> str:
    """B. Tree with sizes + per-TSV raw vs parsed stats."""
    lines = ["## B. Inventory", "", "Tree of `student_resource/` (excluding "
             + ", ".join(f"`{d}`" for d in sorted(SKIP_DIRS)) + "):", "", "```"]
    for root, dirs, files in os.walk(PROJECT_ROOT):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        rel = Path(root).relative_to(PROJECT_ROOT)
        depth = 0 if str(rel) == "." else len(rel.parts)
        lines.append("    " * depth + (f"{rel.name}/" if depth else "student_resource/"))
        for fn in sorted(files):
            size = (Path(root) / fn).stat().st_size
            lines.append("    " * (depth + 1) + f"{fn}  [{human(size)}]")
    lines += ["```", "", "Raw line count (incl. header) vs rows parsed by `io_utils.read_tsv`:", ""]
    rows = []
    for rel, sc in scans.items():
        parsed = parsed_rows.get(rel)
        rows.append([rel, f"{sc['lines']:,}", f"{parsed:,}" if parsed is not None else "n/a (--limit)",
                     ("OK" if parsed == sc["lines"] - 1 - sc["blank"] else "MISMATCH") if parsed is not None else "-",
                     sc["blank"], sc["cr"], sc["bom"], sc["trailing_newline"],
                     sc["bad_tabs"] or "none", f"{sc['quote_lines']:,}"])
    lines.append(md_table(["File", "Raw lines", "Parsed rows", "lines-1-blank == rows?", "Blank lines",
                           "Lines with \\r", "UTF-8 BOM", "Ends with \\n",
                           "Lines with unexpected tab count {tabs: n}", "Lines containing \""], rows))
    lines.append("")
    for rel, sc in scans.items():
        lines.append(f"**{rel}** columns: `{sc['columns']}`; first 3 raw lines (`repr` of decoded UTF-8):")
        lines += ["```", *[repr(b.decode("utf-8", errors="replace")) for b in sc["first"]], "```"]
    return "\n".join(lines)


VALIDATOR_CHECKS = [
    ["ERROR", "`test_source1.tsv` missing in `--test-dir` (validation stops)"],
    ["ERROR", "matching file not found / empty file"],
    ["ERROR", "header has no TAB but contains commas (file looks like CSV)"],
    ["ERROR", "header != `source1_entity_id, matched_entity_ids` (candidate: `source1_entity_id, "
              "candidate_entity_ids`); compared after strip + lower-case"],
    ["ERROR", "non-blank row with no TAB (malformed row)"],
    ["ERROR", "duplicate `source1_entity_id` rows"],
    ["ERROR", "the same ID repeated inside one ID list"],
    ["ERROR", "S1- ids inside an ID list (self-matches)"],
    ["ERROR", "ids without an `S2-`/`S3-` prefix (list split on `,` with NO whitespace strip, so "
              "`S2-1, S3-2` fails on `' S3-2'`)"],
    ["ERROR (only with `--check-ids`)", "ids not present in test_source2/3.tsv"],
    ["ERROR", "required test S1 ids missing from the file"],
    ["ERROR", "rows whose S1 id is not in the test set (row S1 id is NOT stripped; required ids ARE)"],
    ["ERROR", "file not valid UTF-8 / unreadable (OSError)"],
    ["(same rules)", "all of the above are applied to candidate_pairs.tsv when that file exists"],
    ["WARNING", "ID-existence check is OFF (default; enable with `--check-ids`)"],
    ["WARNING", "`--check-ids` given but test_source2/3.tsv missing -> existence check skipped"],
    ["WARNING", "candidate file not found (default path `output/candidate_pairs.tsv`)"],
    ["WARNING", "matched ids not contained in that S1's candidate list"],
]

README_SUMMARY = [
    "Task: for every Source-1 (deduplicated reference) business, list all matching Source-2/Source-3 records; 0..many matches.",
    "Data: TSV files (entity_id, business_name, business_address, country); train = US + India with ground truth, test adds unseen France.",
    "Output: `matching_results.tsv` (scored) and `candidate_pairs.tsv` (the last candidate set actually fed to the model; matches should be a subset).",
    "Metric: macro F0.5 per S1 entity (singletons included: empty-vs-empty = 1, any false merge on a singleton = 0); precision weighted 2x.",
    "Rules: no external data/APIs/geocoding; final model MIT/Apache-2.0 and <= 8B params; submit zip with output/, code/business_entity_resolution/ (src, README, requirements) and the filled Documentation_template.md.",
]


def section_docs() -> str:
    """C. Provided docs (validator checks + CLI, README summary)."""
    help_txt = subprocess.run([sys.executable, "utils/validate_submission.py", "--help"],
                              cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8").stdout
    return "\n".join([
        "## C. Provided docs", "",
        "### utils/validate_submission.py — checks", "",
        "Exit code 0 = PASS, 1 = at least one ERROR (warnings never fail). Stdlib only.", "",
        md_table(["Severity", "Check"], VALIDATOR_CHECKS), "",
        "Other behaviour: blank lines are ignored; an empty ID list is a valid row; the ID list is everything "
        "after the first TAB with only `\\n` stripped (a CRLF file would leave `\\r` on the last id).", "",
        "CLI (`--help`, captured live):", "", "```", help_txt.rstrip(), "```", "",
        "### README.md — 5-line summary", "",
        *[f"{i}. {s}" for i, s in enumerate(README_SUMMARY, 1)], "",
        "Full text of `Documentation_template.md` is in the Appendix.",
    ])


# ------------------------------------------------------------------ section D / E / F
MOJIBAKE_RE = r"[ÂÃ][\s\x{0080}-\x{00BF}]|â€"  # UTF-8 read as Latin-1, e.g. "Â\x80"
NULL_RE = r"(?i)(^|[^a-z])null([^a-z]|$)"


def integrity(label: str, sources: dict[str, dict], rng) -> list[str]:
    """D. Per-file integrity checks for one split's source files."""
    out = []
    for src, d in sources.items():
        ids = d["id"]
        n = len(ids)
        bad_re = ~regex_mask(ids, r"^S[123]-\d+$")
        bad_pfx = ~to_np_bool(pc.starts_with(ids, f"{src}-"))
        dup_rows = sum(c for _, c in value_counts(ids) if c > 1)
        rows = [["ID regex `^S[123]-\\d+$` violations", pct(bad_re.sum(), n)],
                [f"IDs without the file's own prefix `{src}-`", pct(bad_pfx.sum(), n)],
                ["Rows whose entity_id is duplicated", pct(dup_rows, n)]]
        for col, short in COLS.items():
            a = d[short]
            rows.append([f"`{col}` empty", pct((~nonempty(a)).sum(), n)])
            rows.append([f"`{col}` has leading/trailing whitespace",
                         pct(to_np_bool(pc.not_equal(pc.utf8_trim_whitespace(a), a)).sum(), n)])
            rows.append([f"`{col}` contains `\"`", pct(to_np_bool(pc.match_substring(a, '"')).sum(), n)])
            if short in ("name", "addr"):
                rows.append([f"`{col}` contains a `null` placeholder token (any case, e.g. `<NULL>`)",
                             pct(regex_mask(a, NULL_RE).sum(), n)])
                rows.append([f"`{col}` contains mojibake (`Â`/`Ã` + space or U+0080–U+00BF, or `â€`)",
                             pct(regex_mask(a, MOJIBAKE_RE).sum(), n)])
        out += [f"#### {label}_source{src[1]}.tsv ({n:,} rows)", "", md_table(["Check", "Count"], rows), ""]
        if bad_re.any():
            out += ["ID violations (up to 5): " + ", ".join(repr(x) for x in filt(ids, bad_re).to_pylist()[:5]), ""]

        country = cat_np(d["country"])
        countries = sorted(set(country))
        nrows, examples, srows, ex = [], [], [], []
        script_counts: dict = defaultdict(Counter)  # (col, country) -> script -> #fields
        script_examples: dict = defaultdict(list)   # script -> [(col, row)]
        for col in ("business_name", "business_address"):
            a = d[COLS[col]]
            mask = regex_mask(a, r"[^\x00-\x7f]")
            for c in countries:
                cm = country == c
                nrows.append([col, c, pct((mask & cm).sum(), cm.sum())])
            idx = np.flatnonzero(mask)
            for i in rng.choice(idx, size=min(5, len(idx)), replace=False) if len(idx) else []:
                examples.append([col, ids[int(i)].as_py(), a[int(i)].as_py(), country[i]])
            for i, v in zip(idx, filt(a, mask).to_pylist()):
                for s in {script_of(ch) for ch in set(v) if ord(ch) > 127}:
                    script_counts[(col, country[i])][s] += 1
                    if s not in ("LATIN", "non-letter symbol/punct") and len(script_examples[s]) < 400:
                        script_examples[s].append((col, int(i)))
        out += ["Fields with non-ASCII characters, by country:", "",
                md_table(["Column", "Country", "Fields with non-ASCII"], nrows), ""]
        if examples:
            out += ["Random non-ASCII examples (up to 5 per column):", "",
                    md_table(["Column", "entity_id", "Value", "Country"], examples), ""]
        srows = [[col, c, s, f"{k:,}"] for (col, c), cnt in sorted(script_counts.items())
                 for s, k in cnt.most_common()]
        if srows:
            out += ["Scripts among non-ASCII chars (number of fields containing >=1 char of that script; "
                    "label = first word of the Unicode character name, letters/marks only):", "",
                    md_table(["Column", "Country", "Script", "Fields"], srows), ""]
        for s in sorted(script_examples):
            cand = script_examples[s]
            for j in rng.choice(len(cand), size=min(3, len(cand)), replace=False):
                col, i = cand[j]
                ex.append([s, col, *rec(d, i)])
        out += (["Non-Latin script examples (up to 3 per script):", "",
                 md_table(["Script", "Found in", "entity_id", "business_name", "business_address", "country"], ex), ""]
                if ex else ["No non-Latin letters found.", ""])
    return out


def gt_integrity(gt_s1: pa.Array, gt_m: pa.Array) -> list[str]:
    """D. Integrity checks for the ground-truth file."""
    n = len(gt_s1)
    flat = pc.list_flatten(pc.split_pattern(gt_m, ","))
    ne = nonempty(flat)
    flat_ne = filt(flat, ne)
    n_single = int((~nonempty(gt_m)).sum())
    dup_rows = sum(c for _, c in value_counts(gt_s1) if c > 1)
    rows = [
        ["source1_entity_id regex `^S1-\\d+$` violations", pct((~regex_mask(gt_s1, r"^S1-\d+$")).sum(), n)],
        ["rows whose source1_entity_id is duplicated", pct(dup_rows, n)],
        ["matched_entity_ids empty (singletons)", pct(n_single, n)],
        ["matched ids total (after split on ',')", f"{len(flat_ne):,}"],
        ["matched ids violating `^S[23]-\\d+$`", pct((~regex_mask(flat_ne, r"^S[23]-\d+$")).sum(), len(flat_ne))],
        ["empty items inside a non-empty list (e.g. `a,,b`)", f"{int((~ne).sum()) - n_single:,}"],
        ["matched ids with surrounding whitespace",
         f"{int(to_np_bool(pc.not_equal(pc.utf8_trim_whitespace(flat_ne), flat_ne)).sum()):,}"],
        ["fields containing `\"`", f"{int(to_np_bool(pc.match_substring(gt_s1, chr(34))).sum() + to_np_bool(pc.match_substring(gt_m, chr(34))).sum()):,}"],
    ]
    return ["#### train_ground_truth.tsv", "", md_table(["Check", "Count"], rows), ""]


def country_counts(label: str, sources: dict[str, dict]) -> list[str]:
    """E. Exact value_counts of `country` per file."""
    out = []
    for src, d in sources.items():
        n = len(d["country"])
        out += [f"**{label}_source{src[1]}.tsv**", "",
                md_table(["country (repr)", "count", "%"],
                         [[repr(k), f"{v:,}", p2(v, n)] for k, v in value_counts(d["country"])]), ""]
    return out


def size_table(split: Split) -> tuple[pd.DataFrame, list[str]]:
    """F. rows per source x country."""
    tab = pd.DataFrame({src: dict(value_counts(d["country"])) for src, d in split.sources.items()})
    tab = tab.fillna(0).astype(int)
    tab["ratio"] = ((tab["S2"] + tab["S3"]) / tab["S1"].replace(0, np.nan)).round(3)
    rows = [[c, *(f"{tab.at[c, s]:,}" for s in ("S1", "S2", "S3")), tab.at[c, "ratio"]] for c in tab.index]
    tot = tab[["S1", "S2", "S3"]].sum()
    rows.append(["**total**", *(f"{tot[s]:,}" for s in ("S1", "S2", "S3")),
                 round((tot["S2"] + tot["S3"]) / tot["S1"], 3)])
    return tab, [f"**{split.name}**", "", md_table(["country", "S1", "S2", "S3", "(S2+S3)/S1"], rows), ""]


# ------------------------------------------------------------------ section G
def build_pairs(split: Split, gt_s1: pa.Array, gt_m: pa.Array) -> dict:
    """Explode the ground truth into (S1 position, record position) pairs.

    Returns dict with numpy ``gt_row``, ``i1`` (S1 row, -1 if unknown), ``ir``
    (row in split.r, -1 if missing) and pyarrow ``rec_id`` (the listed id).
    """
    lists = pc.split_pattern(gt_m, ",")
    flat = pc.list_flatten(lists)
    parent = pc.list_parent_indices(lists).to_numpy()
    keep = nonempty(flat)
    rec_id = filt(flat, keep)
    parent = parent[keep]
    gt_s1_pos = lookup(gt_s1, split.s1["id"])
    return {"gt_row": parent, "i1": gt_s1_pos[parent], "ir": lookup(rec_id, split.r["id"]),
            "rec_id": rec_id, "gt_s1_pos": gt_s1_pos}


def section_gt(split: Split, gt_s1: pa.Array, gt_m: pa.Array, P: dict) -> list[str]:
    """G. Ground-truth statistics."""
    out = ["## G. Ground truth (train)", ""]
    n1 = len(split.s1["id"])
    c1 = split.s1["country_np"]
    # G1 coverage
    vc = pc.value_counts(gt_s1)
    pos = lookup(split.s1["id"], vc.field("values"))
    cnt = np.where(pos >= 0, vc.field("counts").to_numpy()[np.maximum(pos, 0)], 0)
    same_order = int(to_np_bool(pc.equal(gt_s1, split.s1["id"])).sum()) if len(gt_s1) == n1 else None
    out += ["### G1. Row coverage", "", md_table(["Check", "Value"], [
        ["train S1 ids", f"{n1:,}"], ["GT rows", f"{len(gt_s1):,}"],
        ["S1 ids with exactly one GT row", pct((cnt == 1).sum(), n1)],
        ["S1 ids with no GT row (missing)", f"{int((cnt == 0).sum()):,}"],
        ["S1 ids with >1 GT row", f"{int((cnt > 1).sum()):,}"],
        ["GT rows whose S1 id is not in train_source1 (extra)", f"{int((P['gt_s1_pos'] < 0).sum()):,}"],
        ["GT row i has the same S1 id as source1 row i",
         pct(same_order, n1) if same_order is not None else "n/a (different lengths)"],
    ]), ""]

    i1, ir = P["i1"], P["ir"]
    valid = (i1 >= 0) & (ir >= 0)
    src_all = np.where(ir >= 0, split.r["src"][np.maximum(ir, 0)], "?")
    n_s2 = np.bincount(i1[valid & (src_all == "S2")], minlength=n1)
    n_s3 = np.bincount(i1[valid & (src_all == "S3")], minlength=n1)
    n_all = np.bincount(i1[i1 >= 0], minlength=n1)  # includes ids missing from sources
    single = n_all == 0

    log("G: metric sanity checks via metrics.py")
    true_map = load_ground_truth()
    s1_ids = split.s1["id"].to_pylist()
    if len(true_map) != n1:  # --limit debug runs: restrict to the loaded S1 rows
        true_map = {k: true_map.get(k, set()) for k in s1_ids}
    cmap = dict(zip(s1_ids, c1))
    res_empty = macro_f05({}, true_map, cmap)
    res_perfect = macro_f05(true_map, true_map, cmap)
    singleton_rate = sum(1 for v in true_map.values() if not v) / len(true_map)
    del true_map, cmap, s1_ids
    gc.collect()

    rows = []
    for c in [*split.countries, "ALL"]:
        m = np.ones(n1, bool) if c == "ALL" else (c1 == c)
        if not m.any():
            continue
        na = n_all[m]
        buckets = [(na == k).sum() for k in range(6)] + [((na >= 6) & (na <= 10)).sum(), (na > 10).sum()]
        ns = ~single[m]
        rows.append([c, f"{m.sum():,}", pct(single[m].sum(), m.sum()),
                     *[p2(b, m.sum()) for b in buckets], f"{na.mean():.3f}", int(na.max()),
                     f"{n_s2[m].sum():,}", f"{n_s3[m].sum():,}",
                     pct(((n_s2[m] >= 2) | (n_s3[m] >= 2))[ns].sum(), ns.sum())])
    out += ["### G2. Matches per S1", "",
            "Bucket columns = % of S1 with that many matches.", "",
            md_table(["country", "S1", "singletons", "0", "1", "2", "3", "4", "5", "6-10", ">10",
                      "mean", "max", "S2 matches", "S3 matches",
                      "non-singletons with >=2 matches from the SAME source"], rows), "",
            f"Singleton rate from `load_ground_truth()`: {singleton_rate:.6f}", "",
            "Metric sanity (via `metrics.macro_f05`):", "",
            md_table(["Prediction", "overall", "singleton", "non-singleton",
                      *[f"country={c}" for c in res_empty["by_country"]]],
                     [[name, f"{r['overall']:.6f}",
                       f"{r['by_type'].get('singleton', {}).get('score', float('nan')):.6f}",
                       f"{r['by_type'].get('non_singleton', {}).get('score', float('nan')):.6f}",
                       *[f"{r['by_country'][c]['score']:.6f}" for c in r["by_country"]]]
                      for name, r in (("all-empty", res_empty), ("perfect (GT itself)", res_perfect))]), "",
            f"All-empty score equals singleton rate: **{abs(res_empty['overall'] - singleton_rate) < 1e-12}**; "
            f"perfect score == 1.0: **{res_perfect['overall'] == 1.0}**", ""]

    # G3 consistency: one-home / missing / prefixes / cross-country
    rec_vc = pc.value_counts(P["rec_id"])
    multi_ids = pc.filter(rec_vc.field("values"), pc.greater(rec_vc.field("counts"), 1))
    in_multi = to_np_bool(pc.is_in(P["rec_id"], value_set=multi_ids)) if len(multi_ids) else np.zeros(len(i1), bool)
    sub = pd.DataFrame({"id": filt(P["rec_id"], in_multi).to_pylist(), "row": P["gt_row"][in_multi]})
    rows_per_id = sub.groupby("id")["row"].nunique()
    cross_row = rows_per_id[rows_per_id > 1]
    within_dup = int(len(sub) - sub.drop_duplicates().shape[0])
    ex = [[rid, ", ".join(sorted({gt_s1[int(r)].as_py() for r in sub.loc[sub["id"] == rid, "row"]}))]
          for rid in cross_row.index[:5]]
    missing = filt(P["rec_id"], ir < 0)
    bad_prefix = int((~regex_mask(P["rec_id"], r"^S[23]-")).sum())
    cc = np.zeros(len(i1), bool)
    cc[valid] = c1[i1[valid]] != split.r["country_np"][ir[valid]]
    out += ["### G3. Consistency", "", md_table(["Check", "Value"], [
        ["(S1, record) pairs listed in GT", f"{len(i1):,}"],
        ["S2/S3 ids appearing in >1 GT row (one-home violations)", f"{len(cross_row):,}"],
        ["ids repeated within the same GT row", f"{within_dup:,}"],
        ["GT ids missing from train_source2/3", pct(len(missing), len(i1))],
        ["GT ids with wrong prefix (not S2-/S3-)", f"{bad_prefix:,}"],
        ["cross-country matched pairs (S1 country != record country)", pct(cc.sum(), valid.sum())],
    ]), ""]
    if ex:
        out += ["One-home violations (up to 5):", "", md_table(["record id", "GT rows (S1 ids)"], ex), ""]
    if len(missing):
        out += ["Missing ids (up to 5): " + ", ".join(missing.to_pylist()[:5]), ""]
    if cc.any():
        ct = pd.crosstab(c1[i1[cc]], split.r["country_np"][ir[cc]])
        out += ["Cross-country pairs (rows = S1 country, columns = record country):", "",
                md_table(["S1 \\ record", *ct.columns], [[i, *[f"{v:,}" for v in r]] for i, r in ct.iterrows()]), ""]

    # G4 orphans
    matched_rec = np.zeros(len(split.r["src"]), bool)
    matched_rec[ir[ir >= 0]] = True
    rc = split.r["country_np"]
    rows = []
    for src in ("S2", "S3"):
        for c in [*split.countries, "ALL"]:
            m = (split.r["src"] == src) & (True if c == "ALL" else rc == c)
            if m.any():
                rows.append([src, c, f"{m.sum():,}", pct((~matched_rec[m]).sum(), m.sum())])
    out += ["### G4. Orphans (S2/S3 records in no GT row)", "",
            md_table(["source", "country", "records", "orphans"], rows), ""]

    # G5 leak check
    def numeric(ids) -> np.ndarray:
        return pc.cast(pc.utf8_slice_codeunits(ids, 3), pa.int64()).to_numpy(zero_copy_only=False)

    s1_num, r_num = numeric(split.s1["id"]), numeric(split.r["id"])
    rows = []
    for src in ("S2", "S3"):
        m = valid & (src_all == src)
        if m.sum() > 2:
            rho_row = spearmanr(i1[m], split.r["row"][ir[m]]).statistic
            rho_num = spearmanr(s1_num[i1[m]], r_num[ir[m]]).statistic
            rows.append([src, f"{m.sum():,}", f"{rho_row:+.5f}", f"{rho_num:+.5f}"])
    out += ["### G5. Leak check (Spearman correlation over matched pairs)", "",
            md_table(["record source", "pairs", "S1 row index vs record row index",
                      "S1 numeric id vs record numeric id"], rows), ""]
    return out


# ------------------------------------------------------------------ section H / I
def section_pairs(split: Split, P: dict) -> tuple[list[str], list[str]]:
    """H (true-pair diagnostics) and I (token substitutions), one field at a time."""
    ok = (P["i1"] >= 0) & (P["ir"] >= 0)
    i1, ir = P["i1"][ok], P["ir"][ok]
    n = len(i1)
    ctry = split.s1["country_np"][i1]
    src = split.r["src"][ir]
    ctry_list = ctry.tolist()
    log(f"H: {n:,} valid pairs")

    def pair(field):
        return take(split.s1[field], i1), take(split.r[field], ir)

    def token_pass(a, b):
        """Shared-token flag per pair + one-sided token counters per country."""
        sh = np.zeros(n, bool)
        oa, ob = defaultdict(Counter), defaultdict(Counter)
        for s in range(0, n, CHUNK):
            xa, xb = a.slice(s, CHUNK).to_pylist(), b.slice(s, CHUNK).to_pylist()
            for j, (x, y, c) in enumerate(zip(xa, xb, ctry_list[s:s + CHUNK])):
                sx, sy = set(x.split()), set(y.split())
                if sx & sy:
                    sh[s + j] = True
                if sx and sy:  # substitutions only when both sides are non-empty
                    oa[c].update(sx - sy)
                    ob[c].update(sy - sx)
        return sh, oa, ob

    def buckets(x, y):
        both = to_np_bool(pc.and_(pc.is_valid(x), pc.is_valid(y)))
        same = to_np_bool(pc.equal(x, y)) & both
        return same, both & ~same, ~both

    ne, eq, ratio, share, only_a, only_b = {}, {}, {}, {}, {}, {}
    for f in ("nname", "sname", "naddr"):
        log(f"H: field {f}")
        a, b = pair(f)
        ne[f] = (nonempty(a), nonempty(b))
        eq[f] = to_np_bool(pc.equal(a, b)) & ne[f][0] & ne[f][1]
        ratio[f] = tsr(a, b)
        if f in ("nname", "naddr"):
            share[f], only_a[f], only_b[f] = token_pass(a, b)
        if f == "naddr":
            post = buckets(tn.postal_like(a), tn.postal_like(b))
            house = buckets(tn.house_number(a), tn.house_number(b))
            land = regex_mask(b, tn.LANDMARK_RE)
        del a, b
        gc.collect()
    eq["name+addr"] = eq["nname"] & eq["naddr"]
    a, b = pair("name")
    dba_a = regex_mask(pc.utf8_lower(a), tn.DBA_RE)
    dba_b = regex_mask(pc.utf8_lower(b), tn.DBA_RE)
    del a, b

    groups = [(c, s) for c in split.countries for s in ("S2", "S3")] + [("ALL", "S2+S3")]
    t_eq, t_q, t_tok, t_pc, t_misc = [], [], [], [], []
    for c, s in groups:
        m = np.ones(n, bool) if c == "ALL" else (ctry == c)
        if s != "S2+S3":
            m &= src == s
        k = int(m.sum())
        if not k:
            continue
        both_name = ne["nname"][0][m] & ne["nname"][1][m]
        both_addr = ne["naddr"][0][m] & ne["naddr"][1][m]
        t_eq.append([c, s, f"{k:,}", p2(eq["nname"][m].sum(), k), p2(eq["sname"][m].sum(), k),
                     p2(eq["naddr"][m].sum(), k), p2(eq["name+addr"][m].sum(), k),
                     p2((~ne["naddr"][0][m]).sum(), k), p2((~ne["naddr"][1][m]).sum(), k),
                     p2((~ne["nname"][1][m]).sum(), k), p2((~ne["sname"][1][m]).sum(), k)])
        for f, lab in (("nname", "normalized name"), ("sname", "stripped name"), ("naddr", "normalized address")):
            t_q.append([c, s, lab, *quantiles(ratio[f][m], [1, 5, 10, 25, 50])])
        t_tok.append([c, s, p2(share["nname"][m].sum(), k), p2(share["nname"][m][both_name].sum(), both_name.sum()),
                      p2(share["naddr"][m].sum(), k), p2(share["naddr"][m][both_addr].sum(), both_addr.sum())])
        t_pc.append([c, s, "postal-like (5-6 digits)", *(p2(x[m].sum(), k) for x in post)])
        t_pc.append([c, s, "house number (first 1-4 digits)", *(p2(x[m].sum(), k) for x in house)])
        t_misc.append([c, s, p2(land[m].sum(), k), p2(dba_a[m].sum(), k), p2(dba_b[m].sum(), k),
                       p2((dba_a[m] | dba_b[m]).sum(), k)])

    H = ["## H. True-pair diagnostics (train, all S1–match pairs)", "",
         "Grouped by the S1 record's country x the matched record's source. Exact equality requires both "
         "sides non-empty (empty == empty is NOT counted as equal).", "",
         "### H1. Exact equality", "",
         md_table(["country", "source", "pairs", "=normalized name", "=stripped name", "=normalized address",
                   "=name AND address", "S1 address empty", "noisy address empty", "noisy name empty",
                   "noisy stripped name empty"], t_eq), "",
         "### H2. rapidfuzz token_set_ratio quantiles (all pairs; an empty side scores 0)", "",
         md_table(["country", "source", "field", "q1", "q5", "q10", "q25", "q50"], t_q), "",
         "### H3. Token overlap", "",
         md_table(["country", "source", ">=1 shared name token (all pairs)", "... (both names non-empty)",
                   ">=1 shared address token (all pairs)", "... (both addresses non-empty)"], t_tok), "",
         "### H4. Postal-like code and house number (from normalized addresses)", "",
         "Postal-like = the LAST standalone 5-6 digit run after joining `ddd ddd`. Caveat: US street numbers "
         "can be 5 digits, so for US this rule also catches house numbers.", "",
         md_table(["country", "source", "code", "both present & equal", "both present & different",
                   "missing on either side"], t_pc), "",
         "### H5. Landmarks and trade-name markers", "",
         "Landmark words are matched as whole tokens on the noisy (S2/S3) normalized address. Trade-name markers "
         f"are matched on the lower-cased raw name with `{tn.DBA_RE}`.", "",
         md_table(["country", "source", "noisy address has landmark", "S1 name has marker",
                   "noisy name has marker", "either name has marker"], t_misc), ""]

    I = ["## I. Token substitutions in true pairs", "",
         "Count = number of pairs in which the token appears on that side only (normalized text; pairs where "
         "either side of that field is empty are skipped). S2 and S3 are pooled as the noisy side.", ""]
    for c in split.countries:
        if c not in only_a["nname"]:
            continue
        I += [f"### {c}", ""]
        for k, lab in (("nname", "name"), ("naddr", "address")):
            for side, cnt in (("S1-only", only_a[k][c]), ("noisy-only", only_b[k][c])):
                I.append(f"- **{lab}, {side}:** " + ", ".join(f"{t}({v:,})" for t, v in cnt.most_common(25)))
        I.append("")
    return H, I


# ------------------------------------------------------------------ section J
def s1_name_collisions(split: Split, rng) -> list[str]:
    """S1 records sharing (country, stripped name): duplicate names in the reference source."""
    sname = split.s1["sname"]
    c1 = split.s1["country_np"]
    v = nonempty(sname)
    rows, ex = [], []
    for c in split.countries:
        cm = c1 == c
        n_c = int(cm.sum())
        if not n_c:
            continue
        m = cm & v
        vc = pc.value_counts(filt(sname, m))
        counts = vc.field("counts").to_numpy()
        coll = counts >= 2
        rows.append([split.name, c, f"{n_c:,}", f"{int(m.sum()):,}", f"{int(coll.sum()):,}",
                     pct(counts[coll].sum(), n_c), int(counts.max()) if len(counts) else 0])
        names = vc.field("values").to_numpy(zero_copy_only=False)[coll]
        if len(names):
            chosen = list(rng.choice(names, size=min(5, len(names)), replace=False))
            hit = np.flatnonzero(m & to_np_bool(pc.is_in(sname, value_set=pa.array(chosen, LARGE))))
            by_name = defaultdict(list)
            for p, nm in zip(hit, take(sname, hit).to_pylist()):
                by_name[nm].append(p)
            for nm in chosen:
                for p in by_name[nm][:6]:
                    ex.append([c, nm, len(by_name[nm]), *rec(split.s1, p)[:3]])
    return [md_table(["split", "country", "S1 rows", "non-empty stripped name", "colliding names (>=2 S1)",
                      "S1 rows affected", "max repeats"], rows), "",
            f"Examples ({split.name}; 5 random colliding names per country, up to 6 records each):", "",
            md_table(["country", "stripped name", "#S1 with it", "entity_id", "business_name",
                      "business_address"], ex), ""]


def collision_negatives(split: Split, P: dict, field: str, other: str, rng, n_examples: int) -> list[str]:
    """Non-matching (S1, record) pairs that share (country, <field>) exactly.

    Counts are exact (from key frequencies). Examples and quantiles use a
    Bernoulli sample of affected S1 rows sized to ~NEG_SAMPLE_TARGET pairs, which
    keeps every negative pair's inclusion probability equal.
    """
    label = {"sname": "stripped name", "naddr": "normalized address"}[field]
    olab = {"naddr": "ADDRESS", "nname": "NAME"}[other]
    sep = pa.scalar("\x1f", LARGE)
    key1 = pc.binary_join_element_wise(split.s1["country"], split.s1[field], sep)
    keyr = pc.binary_join_element_wise(split.r["country"], split.r[field], sep)
    v1, vr = nonempty(split.s1[field]), nonempty(split.r[field])
    vc = pc.value_counts(filt(keyr, vr))
    pos = lookup(key1, vc.field("values"))
    n_same = np.where((pos >= 0) & v1, vc.field("counts").to_numpy()[np.maximum(pos, 0)], 0).astype(np.int64)
    del vc, pos

    ok = (P["i1"] >= 0) & (P["ir"] >= 0)
    i1v, irv = P["i1"][ok], P["ir"][ok]
    tp = v1[i1v] & to_np_bool(pc.equal(take(key1, i1v), take(keyr, irv)))
    true_same = np.bincount(i1v[tp], minlength=len(n_same))
    neg = n_same - true_same
    total_neg, total_true = int(neg.sum()), int(tp.sum())
    n1 = len(n_same)
    c1 = split.s1["country_np"]
    out = [md_table(["Metric", "Value"], [
        [f"(S1, record) pairs with identical {label}, same country, NOT a true match", f"{total_neg:,}"],
        ["S1 records affected (>=1 such non-match)", pct((neg > 0).sum(), n1)],
        [f"true pairs with identical {label} (same country)", f"{total_true:,}"],
        [f"precision of the rule 'identical {label} + same country => match'",
         p2(total_true, total_true + total_neg)],
    ]), "", "Per S1 country:", "",
        md_table(["country", "non-match pairs", "S1 affected", "true pairs sharing it", "rule precision"],
                 [[c, f"{int(neg[m].sum()):,}", pct((neg[m] > 0).sum(), m.sum()), f"{int(true_same[m].sum()):,}",
                   p2(true_same[m].sum(), true_same[m].sum() + neg[m].sum())]
                  for c in split.countries if (m := c1 == c).any()]), ""]
    if total_neg == 0:
        return out

    # --- Bernoulli sample of affected S1 rows, then join to records with the same key
    p = min(1.0, NEG_SAMPLE_TARGET / total_neg)
    aff = np.flatnonzero(neg > 0)
    sel = aff[rng.random(len(aff)) < p]
    k_sel = take(key1, sel)
    rpos = np.flatnonzero(vr & to_np_bool(pc.is_in(keyr, value_set=pc.unique(k_sel))))
    d1 = pd.DataFrame({"i1": sel, "key": k_sel.to_pylist()})
    dr = pd.DataFrame({"ir": rpos, "key": take(keyr, rpos).to_pylist()})
    m = d1.merge(dr, on="key")[["i1", "ir"]]
    del d1, dr, k_sel, key1, keyr
    nr = len(split.r["src"])
    code = m["i1"].to_numpy().astype(np.int64) * nr + m["ir"].to_numpy()
    m = m[~np.isin(code, i1v.astype(np.int64) * nr + irv)].reset_index(drop=True)
    if len(m) > NEG_SAMPLE_TARGET:
        m = m.iloc[np.sort(rng.choice(len(m), NEG_SAMPLE_TARGET, replace=False))].reset_index(drop=True)
    out += [f"Sampled {len(m):,} of these non-match pairs (Bernoulli rate {p:.4f} over affected S1 rows, seed 42).", ""]

    mi1, mir = m["i1"].to_numpy(), m["ir"].to_numpy()
    a, b = take(split.s1[other], mi1), take(split.r[other], mir)
    both = nonempty(a) & nonempty(b)
    r_neg = tsr(filt(a, both), filt(b, both))
    tpi1, tpir = i1v[tp], irv[tp]
    ra, rb = take(split.s1[other], tpi1), take(split.r[other], tpir)
    both_t = nonempty(ra) & nonempty(rb)
    r_pos = tsr(filt(ra, both_t), filt(rb, both_t))
    out += [f"token_set_ratio of normalized {olab} (pairs with both sides non-empty):", "",
            md_table(["pairs", "n", "q50", "q75", "q90", "q95", "q99"], [
                [f"non-matches sharing {label} (sample)", f"{len(r_neg):,}", *quantiles(r_neg, [50, 75, 90, 95, 99])],
                [f"TRUE matches sharing {label} (all, for contrast)", f"{len(r_pos):,}",
                 *quantiles(r_pos, [50, 75, 90, 95, 99])]]), ""]

    # --- examples with owner status of the record
    pick = rng.choice(len(m), size=min(n_examples, len(m)), replace=False)
    owner_mask = np.isin(P["ir"], mir[pick])
    owners = defaultdict(list)
    for r_, s_ in zip(P["ir"][owner_mask], P["i1"][owner_mask]):
        owners[int(r_)].append(split.s1["id"][int(s_)].as_py() if s_ >= 0 else "?")
    rows = []
    for j in pick:
        s_, r_ = int(mi1[j]), int(mir[j])
        own = owners.get(r_)
        rows.append([*rec(split.s1, s_)[:3], *rec(split.r, r_), "orphan" if not own else "belongs to " + ", ".join(own)])
    out += [f"{len(rows)} random examples:", "",
            md_table(["S1 id", "S1 name", "S1 address", "record id", "record name", "record address",
                      "country", "record status"], rows), ""]
    return out


# ------------------------------------------------------------------ section K
HDR = ["role", "entity_id", "business_name", "business_address", "country"]


def section_samples_train(split: Split, P: dict, rng) -> list[str]:
    """K (train part): matched groups, singletons, orphans per country."""
    out = ["### Train", ""]
    i1, ir = P["i1"], P["ir"]
    n1 = len(split.s1["id"])
    n_all = np.bincount(i1[i1 >= 0], minlength=n1)
    matched_rec = np.zeros(len(split.r["src"]), bool)
    matched_rec[ir[ir >= 0]] = True
    order = np.argsort(i1, kind="stable")
    si1 = i1[order]
    for c in split.countries:
        cm = split.s1["country_np"] == c
        if not cm.any():
            continue
        groups = np.flatnonzero(cm & (n_all > 0))
        rows = []
        for g, s in enumerate(rng.choice(groups, size=min(8, len(groups)), replace=False), 1):
            rows.append([f"group {g}: S1", *rec(split.s1, s)])
            for k in order[np.searchsorted(si1, s):np.searchsorted(si1, s, side="right")]:
                r = ir[k]
                rows.append([f"group {g}: {split.r['src'][r]}", *rec(split.r, r)] if r >= 0 else
                            [f"group {g}: ?", P["rec_id"][int(k)].as_py(), "(missing from sources)", "", ""])
        out += [f"#### {c}: {min(8, len(groups))} matched groups", "", md_table(HDR, rows), ""]
        sing = np.flatnonzero(cm & (n_all == 0))
        rows = [["S1 singleton", *rec(split.s1, s)] for s in rng.choice(sing, size=min(4, len(sing)), replace=False)]
        out += [f"#### {c}: {len(rows)} singleton S1", "", md_table(HDR, rows), ""]
        orph = np.flatnonzero((split.r["country_np"] == c) & ~matched_rec)
        rows = [[f"orphan {split.r['src'][r]}", *rec(split.r, r)]
                for r in rng.choice(orph, size=min(4, len(orph)), replace=False)]
        out += [f"#### {c}: {len(rows)} orphan S2/S3", "", md_table(HDR, rows), ""]
    return out


def section_samples_unseen(test: Split, train_countries: set, rng) -> list[str]:
    """K (test part): samples of every test country not present in train."""
    out = ["### Test — countries not present in train", ""]
    for c in test.countries:
        if c in train_countries:
            continue
        for d, lab, k in ((test.s1, "S1", 12), (test.r, "S2", 8), (test.r, "S3", 8)):
            m = d["country_np"] == c
            if lab != "S1":
                m &= d["src"] == lab
            idx = np.flatnonzero(m)
            rows = [[lab, *rec(d, i)] for i in rng.choice(idx, size=min(k, len(idx)), replace=False)]
            out += [f"#### {c}: {len(rows)} {lab}", "", md_table(HDR, rows), ""]
    return out


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description="EDA -> reports/eda_report.md")
    ap.add_argument("--limit", type=int, default=None, help="debug: only first N rows of each source file")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    S: dict[str, list[str]] = defaultdict(list)

    log("A/B/C")
    S["A"] = [section_machine()]
    scans = {p.relative_to(PROJECT_ROOT).as_posix(): raw_scan(p)
             for split in ("train", "test") for p in sorted((DATASET_DIR / split).glob("*.tsv"))}
    S["C"] = [section_docs()]
    parsed_rows: dict[str, int | None] = {}

    # ---------------- train
    train = Split("train", args.limit)
    for src, d in train.sources.items():
        parsed_rows[f"dataset/train/train_source{src[1]}.tsv"] = None if args.limit else len(d["id"])
    gt_df = read_tsv(DATASET_DIR / "train" / "train_ground_truth.tsv")
    parsed_rows["dataset/train/train_ground_truth.tsv"] = len(gt_df)
    gt_s1, gt_m = arrow(gt_df["source1_entity_id"]), arrow(gt_df["matched_entity_ids"])
    del gt_df
    if args.limit:
        keep = to_np_bool(pc.is_in(gt_s1, value_set=train.s1["id"]))
        gt_s1, gt_m = filt(gt_s1, keep), filt(gt_m, keep)
    log("D/E/F train")
    S["D"] += ["### Train", ""] + integrity("train", train.sources, rng) + gt_integrity(gt_s1, gt_m)
    S["E"] += country_counts("train", train.sources)
    tab_train, lines = size_table(train)
    S["F"] += lines
    log("G")
    P = build_pairs(train, gt_s1, gt_m)
    S["G"] = section_gt(train, gt_s1, gt_m, P)
    log("H/I")
    S["H"], S["I"] = section_pairs(train, P)
    gc.collect()
    log("J train")
    S["J"] += ["### J1. Stripped-name collisions within train S1 (key = country + stripped name)", ""]
    S["J"] += s1_name_collisions(train, rng)
    S["J"] += ["### J2. Train non-match pairs with identical stripped name (same country)", ""]
    S["J"] += collision_negatives(train, P, "sname", "naddr", rng, 10)
    log("J train address")
    S["J"] += ["### J3. Train non-match pairs with identical normalized address (same country)", ""]
    S["J"] += collision_negatives(train, P, "naddr", "nname", rng, 5)
    log("K train")
    S["K"] += section_samples_train(train, P, rng)
    train_countries = set(train.countries)
    del train, P, gt_s1, gt_m
    gc.collect()

    # ---------------- test
    test = Split("test", args.limit)
    for src, d in test.sources.items():
        parsed_rows[f"dataset/test/test_source{src[1]}.tsv"] = None if args.limit else len(d["id"])
    log("D/E/F test")
    S["D"] += ["### Test", ""] + integrity("test", test.sources, rng)
    S["E"] += country_counts("test", test.sources)
    tab_test, lines = size_table(test)
    S["F"] += lines
    S["F"] += ["(S2+S3)/S1 per country, train vs test:", "",
               md_table(["country", "train", "test"],
                        [[c, tab_train["ratio"].get(c, "—"), tab_test["ratio"].get(c, "—")]
                         for c in sorted(set(tab_train.index) | set(tab_test.index))]), ""]
    unseen = [c for c in tab_test.index if c not in train_countries]
    n_test_s1 = int(tab_test["S1"].sum())
    S["F"] += [f"Test countries not present in train: {', '.join(unseen) or 'none'}.", ""] + [
        f"- {c} share of test S1: {pct(tab_test.at[c, 'S1'], n_test_s1)}" for c in unseen] + [""]
    log("J test")
    S["J"] += ["### J4. Stripped-name collisions within test S1, per test country", ""]
    S["J"] += s1_name_collisions(test, rng)
    log("K test")
    S["K"] += section_samples_unseen(test, train_countries, rng)
    del test
    gc.collect()

    # ---------------- format sanity
    log("format sanity")
    n_rows = write_all_empty(EMPTY_SUB_DIR)
    cmd = [sys.executable, "utils/validate_submission.py",
           "--matching", "submissions/00_all_empty/matching_results.tsv",
           "--candidate", "submissions/00_all_empty/candidate_pairs.tsv", "--test-dir", "dataset/test"]
    vr = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8")
    S["FMT"] = ["## Format sanity: all-empty submission", "",
                f"Wrote {n_rows:,} rows to `submissions/00_all_empty/` (matching_results.tsv, candidate_pairs.tsv).",
                "", "Command (from `student_resource/`):", "", "```",
                "python utils/validate_submission.py --matching submissions/00_all_empty/matching_results.tsv "
                "--candidate submissions/00_all_empty/candidate_pairs.tsv --test-dir dataset/test", "```", "",
                f"Exit code: **{vr.returncode}**. Full output:", "", "```", (vr.stdout + vr.stderr).rstrip(), "```"]
    S["B"] = [section_inventory(parsed_rows, scans)]

    # ---------------- assemble
    elapsed = time.time() - T0
    doc = (PROJECT_ROOT / "Documentation_template.md").read_text(encoding="utf-8")
    parts = [
        "# EDA report — Business Entity Resolution", "",
        f"Generated by `code/business_entity_resolution/src/eda.py` in {elapsed / 60:.1f} min"
        + (f" (**DEBUG --limit {args.limit}**)" if args.limit else "") + ". Seed 42 for all sampling.", "",
        "Normalization: lower-case -> NFKD -> drop combining diacritics U+0300–U+036F -> Unicode punctuation/"
        "symbols/whitespace -> single space -> trim (non-Latin scripts are kept). Stripped name = normalized "
        f"name minus tokens `{' '.join(tn.LEGAL_TOKENS)}`. Empty strings are skipped in all collision stats.", "",
        *S["A"], "", *S["B"], "", *S["C"], "",
        "## D. Integrity", "", *S["D"], "",
        "## E. Country labels", "", *S["E"], "",
        "## F. Sizes", "", *S["F"], "",
        *S["G"], "", *S["H"], "", *S["I"], "",
        "## J. Hard negatives", "", *S["J"], "",
        "## K. Raw samples (full text)", "", *S["K"], "",
        *S["FMT"], "",
        "## Appendix: Documentation_template.md (full text)", "", "````markdown", doc.rstrip(), "````", "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(parts), encoding="utf-8")
    log(f"wrote {REPORT_PATH} ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
