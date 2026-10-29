"""카운팅 결과를 수작업 카운트와 비교한다.

- 대수 오차: 전체 · 차로별 · 차종 그룹별 절대 백분율 오차(APE)
- 구간 오차: bin_seconds 단위로 나눈 구간별 MAE · MAPE
- 이벤트 매칭: 같은 차로에서 시각 차이가 tol 이내인 쌍을 헝가리안으로 매칭 → 정밀도 · 재현율 · F1.
  대수가 맞아도 놓친 차와 헛카운트가 상쇄된 경우를 드러낸다.

    python -m src.eval.count_accuracy --manual data/annotations/traffic_mp4_manual.csv \
        --pred results/counting/traffic_mp4_C1_events.csv results/counting/traffic_mp4_C2_events.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

CLASS_GROUP = {
    "car": "small", "motorcycle": "small", "small": "small", "vehicle": "small",
    "van": "large", "truck": "large", "bus": "large", "large": "large",
}


def _ape(gt: int, pred: int) -> float | None:
    return None if gt == 0 else round(abs(pred - gt) / gt * 100, 2)


def match_events(gt: pd.DataFrame, pred: pd.DataFrame, tol: float) -> list[tuple[int, int]]:
    """(gt 행 번호, pred 행 번호) 쌍. 같은 차로 안에서만, |Δt| ≤ tol."""
    pairs: list[tuple[int, int]] = []
    for lane in set(gt["lane"]) & set(pred["lane"]):
        gi = np.flatnonzero(gt["lane"].to_numpy() == lane)
        pi = np.flatnonzero(pred["lane"].to_numpy() == lane)
        dt = np.abs(gt["seconds"].to_numpy()[gi][:, None] - pred["seconds"].to_numpy()[pi][None, :])
        cost = np.where(dt <= tol, dt, 1e6)
        rows, cols = linear_sum_assignment(cost)
        pairs += [(int(gi[r]), int(pi[c])) for r, c in zip(rows, cols) if cost[r, c] <= tol]
    return sorted(pairs)


def evaluate(
    gt: pd.DataFrame,
    pred: pd.DataFrame,
    tol: float = 1.0,
    bin_seconds: float = 10.0,
    start: float = 0.0,
    end: float | None = None,
) -> dict:
    end = end if end is not None else max(gt["seconds"].max(), pred["seconds"].max() if len(pred) else 0) + 1e-6
    in_window = lambda df: df[(df["seconds"] >= start) & (df["seconds"] < end)].reset_index(drop=True)
    gt, pred = in_window(gt), in_window(pred)
    gt = gt.assign(group=gt["cls"].map(CLASS_GROUP))
    pred = pred.assign(group=pred["cls"].map(CLASS_GROUP))

    res: dict = {"window": [start, round(end, 2)]}
    res["total"] = {"gt": len(gt), "pred": len(pred), "ape": _ape(len(gt), len(pred))}
    res["by_lane"] = {
        lane: {"gt": int((gt["lane"] == lane).sum()), "pred": int((pred["lane"] == lane).sum()),
               "ape": _ape(int((gt["lane"] == lane).sum()), int((pred["lane"] == lane).sum()))}
        for lane in sorted(set(gt["lane"]) | set(pred["lane"]))
    }
    res["by_group"] = {
        g: {"gt": int((gt["group"] == g).sum()), "pred": int((pred["group"] == g).sum()),
            "ape": _ape(int((gt["group"] == g).sum()), int((pred["group"] == g).sum()))}
        for g in ["small", "large"]
    }

    edges = np.arange(start, end + bin_seconds, bin_seconds)
    g_bins = np.histogram(gt["seconds"], bins=edges)[0]
    p_bins = np.histogram(pred["seconds"], bins=edges)[0]
    nz = g_bins > 0
    res["bins"] = {
        "seconds": bin_seconds,
        "gt": g_bins.tolist(),
        "pred": p_bins.tolist(),
        "mae": round(float(np.abs(g_bins - p_bins).mean()), 3),
        "mape": round(float((np.abs(g_bins - p_bins)[nz] / g_bins[nz]).mean() * 100), 2) if nz.any() else None,
    }

    pairs = match_events(gt, pred, tol)
    tp = len(pairs)
    fp, fn = len(pred) - tp, len(gt) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    same_group = [gt.loc[g, "group"] == pred.loc[p, "group"] for g, p in pairs]
    has_dir = "direction" in gt and "direction" in pred
    same_dir = [gt.loc[g, "direction"] == pred.loc[p, "direction"] for g, p in pairs] if has_dir else []
    res["events"] = {
        "tol_seconds": tol, "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
        "class_acc": round(float(np.mean(same_group)), 3) if pairs else None,
        "direction_acc": round(float(np.mean(same_dir)), 3) if same_dir else None,
        "mean_abs_dt": round(float(np.mean([abs(gt.loc[g, "seconds"] - pred.loc[p, "seconds"]) for g, p in pairs])), 3) if pairs else None,
        "missed_seconds": sorted(gt.drop(index=[g for g, _ in pairs])["seconds"].round(2).tolist()),
        "false_seconds": sorted(pred.drop(index=[p for _, p in pairs])["seconds"].round(2).tolist()),
    }
    return res


def to_markdown(results: dict[str, dict]) -> str:
    lines = [
        "| 실험 | 수작업 | 예측 | 전체 오차(%) | 10초 구간 MAPE(%) | 정밀도 | 재현율 | F1 | 차종 그룹 정확도 | 방향 정확도 |",
        "|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in results.items():
        e = r["events"]
        fmt = lambda v: "-" if v is None else v
        lines.append(
            f"| {name} | {r['total']['gt']} | {r['total']['pred']} | {r['total']['ape']} | "
            f"{r['bins']['mape']} | {e['precision']:.3f} | {e['recall']:.3f} | {e['f1']:.3f} | "
            f"{fmt(e['class_acc'])} | {fmt(e['direction_acc'])} |"
        )
    lines += ["", "| 실험 | " + " | ".join(f"{lane} (수작업/예측)" for lane in next(iter(results.values()))["by_lane"]) + " | large (수작업/예측) |",
              "|:---:|" + "---:|" * (len(next(iter(results.values()))["by_lane"]) + 1)]
    for name, r in results.items():
        cells = [f"{v['gt']}/{v['pred']}" for v in r["by_lane"].values()]
        lg = r["by_group"]["large"]
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {lg['gt']}/{lg['pred']} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manual", required=True)
    ap.add_argument("--pred", nargs="+", required=True)
    ap.add_argument("--tol", type=float, default=1.0)
    ap.add_argument("--bin-seconds", type=float, default=10.0)
    ap.add_argument("--start", type=float, default=1.0, help="배경 모델 워밍업 구간 제외")
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--out", default="results/counting/accuracy")
    args = ap.parse_args()

    gt = pd.read_csv(args.manual)
    end = args.end if args.end is not None else float(np.ceil(gt["seconds"].max()))
    results = {}
    for path in args.pred:
        name = Path(path).stem.replace("_events", "").split("_")[-1]
        results[name] = evaluate(gt, pd.read_csv(path), args.tol, args.bin_seconds, args.start, end)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    md = to_markdown(results)
    out.with_suffix(".md").write_text(md + "\n", encoding="utf-8")
    print(md)
    for name, r in results.items():
        print(name, "missed:", r["events"]["missed_seconds"], "false:", r["events"]["false_seconds"])


if __name__ == "__main__":
    main()
