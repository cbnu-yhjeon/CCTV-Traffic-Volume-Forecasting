import pandas as pd

from src.eval.count_accuracy import evaluate, match_events


def frame(rows):
    return pd.DataFrame(rows, columns=["seconds", "lane", "cls"])


def test_match_events_within_tolerance_per_lane():
    gt = frame([(1.0, "a", "car"), (5.0, "a", "car"), (2.0, "b", "car")])
    pred = frame([(1.3, "a", "small"), (2.2, "a", "small"), (9.0, "b", "large")])
    pairs = match_events(gt, pred, tol=1.0)
    # gt 1.0(a) ↔ pred 1.3(a) 만 매칭. pred 2.2(a)는 5.0과 멀고, 다른 차로의 2.0과는 매칭 안 됨
    assert pairs == [(0, 0)]


def test_evaluate_counts_and_event_scores():
    gt = frame([(1.0, "a", "car"), (3.0, "a", "van"), (2.0, "b", "car"), (8.0, "b", "car")])
    pred = frame([(1.1, "a", "small"), (3.2, "a", "large"), (2.1, "b", "small"), (5.0, "b", "small")])
    res = evaluate(gt, pred, tol=1.0, bin_seconds=5.0, start=0.0, end=10.0)
    assert res["total"]["gt"] == 4 and res["total"]["pred"] == 4
    assert res["total"]["ape"] == 0.0
    assert res["events"]["tp"] == 3 and res["events"]["fp"] == 1 and res["events"]["fn"] == 1
    assert res["events"]["class_acc"] == 1.0
    assert res["by_lane"]["b"] == {"gt": 2, "pred": 2, "ape": 0.0}
    # 0~5초: gt 3, pred 3 / 5~10초: gt 1, pred 1 → 구간 오차 0
    assert res["bins"]["mae"] == 0.0


def test_evaluate_applies_time_window():
    gt = frame([(0.5, "a", "car"), (2.0, "a", "car")])
    pred = frame([(2.0, "a", "small")])
    res = evaluate(gt, pred, tol=1.0, bin_seconds=10.0, start=1.0, end=10.0)
    assert res["total"]["gt"] == 1 and res["events"]["recall"] == 1.0
