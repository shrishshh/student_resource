"""Evaluation metric: macro-averaged F0.5 over Source-1 entities.

Per S1 entity with true set T and predicted set P:

* T and P both empty          -> 1.0
* exactly one of them empty   -> 0.0
* otherwise                   -> 1.25 * TP / (|P| + 0.25 * |T|)

(the last line is the usual F_beta with beta = 0.5 rewritten in counts:
(1 + b^2) TP / ((1 + b^2) TP + b^2 FN + FP) = 1.25 TP / (|P| + 0.25 |T|)).
The macro score averages this over ALL S1 entities in ``true_map``; an S1
missing from the predictions counts as an empty prediction.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping


def entity_f05(pred: Iterable[str], true: Iterable[str]) -> float:
    """F0.5 for a single Source-1 entity.

    Args:
        pred: Predicted matched ids.
        true: True matched ids.
    """
    p, t = set(pred), set(true)
    if not p and not t:
        return 1.0
    if not p or not t:
        return 0.0
    tp = len(p & t)
    return 1.25 * tp / (len(p) + 0.25 * len(t))


def macro_f05(
    pred_map: Mapping[str, Iterable[str]],
    true_map: Mapping[str, Iterable[str]],
    country_map: Mapping[str, str] | None = None,
) -> dict:
    """Macro F0.5 over every S1 entity in ``true_map``, with breakdowns.

    Args:
        pred_map: ``S1 id -> predicted ids``. Missing S1 ids count as empty;
            extra S1 ids (not in ``true_map``) are ignored.
        true_map: ``S1 id -> true ids`` (empty for singletons).
        country_map: Optional ``S1 id -> country`` for a per-country breakdown.

    Returns:
        ``{"overall": float, "n": int,
           "by_type": {"singleton": {"score", "n"}, "non_singleton": {...}},
           "by_country": {country: {"score", "n"}}}``
        (``by_country`` is empty when ``country_map`` is None).
    """
    sums: dict[str, dict[str, float]] = {"type": defaultdict(float), "country": defaultdict(float)}
    counts: dict[str, dict[str, int]] = {"type": defaultdict(int), "country": defaultdict(int)}
    total = 0.0
    for s1, true in true_map.items():
        true = set(true)
        score = entity_f05(pred_map.get(s1, ()), true)
        total += score
        kind = "non_singleton" if true else "singleton"
        sums["type"][kind] += score
        counts["type"][kind] += 1
        if country_map is not None:
            c = country_map.get(s1, "<unknown>")
            sums["country"][c] += score
            counts["country"][c] += 1

    def _summ(key: str) -> dict:
        return {
            k: {"score": sums[key][k] / counts[key][k], "n": counts[key][k]}
            for k in sorted(counts[key])
        }

    n = len(true_map)
    return {
        "overall": total / n if n else 0.0,
        "n": n,
        "by_type": _summ("type"),
        "by_country": _summ("country"),
    }
