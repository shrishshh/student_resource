"""Unit tests for the F0.5 metric (src/metrics.py)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metrics import entity_f05, macro_f05  # noqa: E402


def test_official_example():
    """README example: P = 2/3, R = 1 -> F0.5 = 0.7142857."""
    pred = {"S2-00047", "S2-00193", "S3-00812"}
    true = {"S2-00047", "S3-00812"}
    assert entity_f05(pred, true) == pytest.approx(0.7142857, abs=1e-7)


def test_singleton_empty_prediction_scores_one():
    assert entity_f05(set(), set()) == 1.0


def test_singleton_any_prediction_scores_zero():
    assert entity_f05({"S2-1"}, set()) == 0.0


def test_non_singleton_empty_prediction_scores_zero():
    assert entity_f05(set(), {"S2-1", "S3-2"}) == 0.0


def test_perfect_prediction_scores_one():
    true = {"S2-1", "S3-2", "S3-3"}
    assert entity_f05(set(true), true) == 1.0


def test_macro_missing_predictions_count_as_empty_and_breakdowns():
    true_map = {"S1-a": set(), "S1-b": {"S2-1"}, "S1-c": {"S3-9"}}
    pred_map = {"S1-b": {"S2-1"}}  # S1-a, S1-c missing -> empty
    countries = {"S1-a": "US", "S1-b": "US", "S1-c": "France"}
    res = macro_f05(pred_map, true_map, countries)
    assert res["overall"] == pytest.approx(2 / 3)
    assert res["n"] == 3
    assert res["by_type"]["singleton"] == {"score": 1.0, "n": 1}
    assert res["by_type"]["non_singleton"]["score"] == pytest.approx(0.5)
    assert res["by_country"]["US"]["score"] == 1.0
    assert res["by_country"]["France"]["score"] == 0.0
