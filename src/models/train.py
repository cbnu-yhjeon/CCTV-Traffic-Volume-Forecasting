"""예측 실험 P0~P4.

| ID | 특징 | 모델 |
|----|------|------|
| P0 | 지난주 같은 시각 값 | 계절성 기준선 |
| P1 | 달력 | XGBoost · LightGBM · RF |
| P2 | 달력 + 기상 | 〃 |
| P3 | 달력 + 기상 + 지연값 | 〃 |
| P4 | P3 세 모델의 가중 평균(검증 구간 NNLS) | 앙상블 |

모든 실험은 같은 행(지연값이 모두 있는 행)에서 학습 · 평가한다. 분할은 시간 순서.

    python -m src.models.train --config configs/forecast/uci_metro.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.ensemble import RandomForestRegressor

from src.eval import plots
from src.eval.forecast_metrics import summarize
from src.features.dataset import build_dataset
from src.models.ensemble import blend, fit_weights

EXPERIMENTS = {
    "P1": ["calendar"],
    "P2": ["calendar", "weather"],
    "P3": ["calendar", "weather", "lag"],
}
MODEL_NAMES = ["XGBoost", "LightGBM", "RandomForest"]
# XGBoost · LightGBM의 OpenMP 스레드가 한 프로세스에서 모든 코어를 잡으면 서로 바쁜 대기로 크게 느려진다
N_JOBS = max(1, min(8, (os.cpu_count() or 2) // 2))


def make_model(name: str, seed: int = 42):
    if name == "XGBoost":
        return xgb.XGBRegressor(
            n_estimators=3000, learning_rate=0.03, max_depth=8, min_child_weight=3,
            subsample=0.8, colsample_bytree=0.8, tree_method="hist",
            early_stopping_rounds=200, random_state=seed, n_jobs=N_JOBS,
        )
    if name == "LightGBM":
        return lgb.LGBMRegressor(
            n_estimators=3000, learning_rate=0.03, num_leaves=63, min_child_samples=20,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed,
            n_jobs=1, verbose=-1,  # 다른 작업과 CPU를 나눌 때 멀티스레드가 수십 배 느려져 단일 스레드로 고정
        )
    if name == "RandomForest":
        return RandomForestRegressor(
            n_estimators=400, min_samples_leaf=2, max_features=0.5, random_state=seed, n_jobs=N_JOBS,
        )
    raise ValueError(name)


def fit(name: str, X_tr, y_tr, X_va, y_va):
    model = make_model(name)
    if name == "XGBoost":
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    elif name == "LightGBM":
        model.fit(X_tr, y_tr, eval_X=X_va, eval_y=y_va, callbacks=[lgb.early_stopping(200, verbose=False)])
    else:
        model.fit(X_tr, y_tr)
    return model


def split(df: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    va, te = pd.Timestamp(cfg["split"]["valid_start"]), pd.Timestamp(cfg["split"]["test_start"])
    return {"train": df[df.index < va], "valid": df[(df.index >= va) & (df.index < te)], "test": df[df.index >= te]}


def plot_results(metrics: pd.DataFrame, preds: pd.DataFrame, out: Path, freq: str) -> None:
    import matplotlib.pyplot as plt

    # 1) 테스트 MAE: 실험 × 모델
    test = metrics[metrics["split"] == "test"]
    fig, ax = plt.subplots(figsize=(8, 4))
    order = ["P0", "P1", "P2", "P3", "P4"]
    labels, values, colors = [], [], []
    color_of = {"Seasonal naive": plots.NEUTRAL, "XGBoost": plots.SERIES[0], "LightGBM": plots.SERIES[1],
                "RandomForest": plots.SERIES[2], "Ensemble": plots.SERIES[6]}
    for exp in order:
        for _, r in test[test["experiment"] == exp].iterrows():
            labels.append(exp)
            values.append(r["MAE"])
            colors.append(color_of[r["model"]])
    bars = ax.bar(range(len(values)), values, color=colors, width=0.7, edgecolor=plots.SURFACE, linewidth=2)
    best = int(np.argmin(values))
    ax.bar_label(bars, labels=[f"{v:.0f}" if i in (0, best) else "" for i, v in enumerate(values)],
                 padding=3, color=plots.INK, fontsize=9)
    centers = {e: np.mean([i for i, lab in enumerate(labels) if lab == e]) for e in order}
    ax.set_xticks(list(centers.values()), [
        {"P0": "P0 기준선", "P1": "P1 달력", "P2": "P2 +기상", "P3": "P3 +지연값", "P4": "P4 앙상블"}[e]
        for e in centers], fontsize=10)
    ax.set_ylabel("MAE (대/시간)" if freq == "1h" else "MAE (대/구간)")
    ax.set_title("테스트 구간 MAE: 실험 · 모델별 (낮을수록 좋음)")
    ax.grid(axis="x", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in color_of.values()]
    ax.legend(handles, list(color_of), ncol=5, loc="upper right", fontsize=8, bbox_to_anchor=(1, 1.02))
    plots.save(fig, out / "mae_by_experiment.png")

    # 2) 테스트 구간 한 주: 실제 vs P0 vs P4
    mondays = preds.index[preds.index.dayofweek == 0].normalize().unique()
    week_start = mondays[len(mondays) // 2]
    wk = preds.loc[week_start: week_start + pd.Timedelta("7D") - pd.Timedelta(freq)]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(wk.index, wk["actual"], color=plots.INK, lw=2.2, label="실제")
    ax.plot(wk.index, wk["P0_Seasonal naive"], color=plots.NEUTRAL, lw=1.4, ls="--", label="P0 지난주 같은 시각")
    ax.plot(wk.index, wk["P4_Ensemble"], color=plots.SERIES[0], lw=1.8, label="P4 앙상블")
    ax.set_title(f"테스트 구간 한 주 ({wk.index[0]:%Y-%m-%d} ~)")
    ax.set_ylabel("교통량")
    ax.legend(ncol=3, loc="upper left", fontsize=9)
    fig.autofmt_xdate()
    plots.save(fig, out / "week_actual_vs_pred.png")

    # 3) 시간대별 MAE: P0 vs P4
    hour = preds.index.hour
    err = pd.DataFrame({
        "P0": (preds["actual"] - preds["P0_Seasonal naive"]).abs(),
        "P4": (preds["actual"] - preds["P4_Ensemble"]).abs(),
    }).groupby(hour).mean()
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(err.index, err["P0"], color=plots.NEUTRAL, ls="--", marker="o", ms=4, label="P0 지난주 같은 시각")
    ax.plot(err.index, err["P4"], color=plots.SERIES[0], marker="o", ms=4, label="P4 앙상블")
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlabel("시각")
    ax.set_ylabel("MAE")
    ax.set_title("시간대별 테스트 MAE")
    ax.legend(fontsize=9)
    plots.save(fig, out / "mae_by_hour.png")


def run(cfg: dict) -> pd.DataFrame:
    out = Path(cfg["out"])
    (out / "models").mkdir(parents=True, exist_ok=True)
    df, groups = build_dataset(cfg)
    usable = df["volume"].notna() & df[groups["lag"]].notna().all(axis=1)
    df = df[usable]
    parts = split(df, cfg)
    for k, v in parts.items():
        print(f"{k:5s} {len(v):6d} rows  {v.index.min()} ~ {v.index.max()}")
    floor = cfg.get("mape_floor_ratio", 0.05) * parts["train"]["volume"].mean()

    rows, test_preds, valid_preds = [], {}, {}

    def record(exp: str, model: str, p_va, p_te, extra: dict | None = None) -> None:
        valid_preds[f"{exp}_{model}"] = np.asarray(p_va)
        test_preds[f"{exp}_{model}"] = np.asarray(p_te)
        for name, y, p in [("valid", parts["valid"]["volume"], p_va), ("test", parts["test"]["volume"], p_te)]:
            rows.append({"split": name, "experiment": exp, "model": model, **summarize(y, p, floor), **(extra or {})})

    record("P0", "Seasonal naive", parts["valid"]["lag_7d"], parts["test"]["lag_7d"])

    for exp, gnames in EXPERIMENTS.items():
        cols = [c for g in gnames for c in groups[g]]
        X = {k: v[cols] for k, v in parts.items()}
        y = {k: v["volume"] for k, v in parts.items()}
        for name in MODEL_NAMES:
            t0 = time.perf_counter()
            model = fit(name, X["train"], y["train"], X["valid"], y["valid"])
            secs = time.perf_counter() - t0
            record(exp, name, model.predict(X["valid"]), model.predict(X["test"]), {"fit_seconds": round(secs, 1)})
            if name != "RandomForest":
                joblib.dump(model, out / "models" / f"{exp}_{name}.joblib")
            print(f"{exp} {name:12s} test MAE={rows[-1]['MAE']:.1f} ({secs:.1f}s)")
        (out / "models" / f"{exp}_features.json").write_text(json.dumps(cols))

    weights = fit_weights({m: valid_preds[f"P3_{m}"] for m in MODEL_NAMES}, parts["valid"]["volume"].to_numpy())
    record(
        "P4", "Ensemble",
        blend({m: valid_preds[f"P3_{m}"] for m in MODEL_NAMES}, weights),
        blend({m: test_preds[f"P3_{m}"] for m in MODEL_NAMES}, weights),
    )
    print("P4 weights:", {k: round(v, 3) for k, v in weights.items()}, f"test MAE={rows[-1]['MAE']:.1f}")

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "metrics.csv", index=False)
    preds = pd.DataFrame(test_preds, index=parts["test"].index)
    preds.insert(0, "actual", parts["test"]["volume"])
    preds.to_csv(out / "predictions_test.csv")
    parts["test"].to_pickle(out / "models" / "test_features.pkl")  # SHAP 분석용
    (out / "groups.json").write_text(json.dumps(groups, ensure_ascii=False, indent=2))
    (out / "ensemble_weights.json").write_text(json.dumps(weights, indent=2))

    test = metrics[metrics["split"] == "test"].drop(columns=["split"])
    p0 = test.loc[test["experiment"] == "P0", "MAE"].iloc[0]
    test["vs_P0_MAE_%"] = ((test["MAE"] - p0) / p0 * 100).round(1)
    md = test[["experiment", "model", "MAE", "RMSE", "MAPE", "WAPE", "vs_P0_MAE_%"]].round(2).to_markdown(index=False)
    (out / "metrics_test.md").write_text(
        f"테스트 {parts['test'].index.min():%Y-%m-%d} ~ {parts['test'].index.max():%Y-%m-%d}, {len(parts['test'])}행. "
        f"MAPE는 실제값 < {floor:.0f} 제외.\n\n{md}\n", encoding="utf-8")
    print(md)
    plot_results(metrics, preds, out, cfg["freq"])
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        run(yaml.safe_load(f))


if __name__ == "__main__":
    main()
