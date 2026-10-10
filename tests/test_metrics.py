import pytest

from src.evaluation.metrics import classification_metrics, span_prf


def test_span_prf_exact_match():
    gold = [[(0, 5), (10, 15)], [], [(3, 7)]]
    pred = [[(0, 5)], [(1, 2)], [(3, 7)]]
    m = span_prf(gold, pred)
    assert m["tp"] == 2
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)


def test_span_prf_partial_overlap_is_a_miss():
    assert span_prf([[(0, 10)]], [[(0, 9)]])["f1"] == 0.0


def test_span_prf_empty_predictions():
    m = span_prf([[(0, 1)]], [[]])
    assert m["precision"] == 0.0 and m["f1"] == 0.0


def test_classification_metrics():
    m = classification_metrics(["positive", "negative", "neutral", "positive"],
                               ["positive", "negative", "positive", "positive"])
    assert m["accuracy"] == 0.75
    assert m["per_class"]["neutral"]["recall"] == 0.0
    assert m["confusion_matrix"]["matrix"][1] == [0, 0, 1]
