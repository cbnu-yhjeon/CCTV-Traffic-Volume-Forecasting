"""SHAP으로 교통량 변화 요인 해석.

- P2(달력 + 기상) 모델: 지연값이 없어 기상 효과가 지연값에 흡수되지 않는다 → 요인 해석의 주 대상
- P3(+ 지연값) 모델: 실제 예측에 쓰는 모델에서 요인 그룹 기여도 비교

    python -m src.models.shap_report --config configs/forecast/uci_metro.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import yaml

from src.eval import plots

GROUP_LABEL = {"calendar": "달력", "weather": "기상", "lag": "지연값"}
GROUP_COLOR = {"calendar": plots.SERIES[0], "weather": plots.SERIES[2], "lag": plots.SERIES[1]}


def shap_values(model, X: pd.DataFrame) -> np.ndarray:
    """XGBoost 내장 TreeSHAP(pred_contribs). shap 0.49는 XGBoost 3.x 모델의 base_score를 읽지 못한다."""
    import xgboost as xgb

    contribs = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
    return contribs[:, :-1]  # 마지막 열은 기준값(bias)


def group_importance(sv: np.ndarray, cols: list[str], groups: dict[str, list[str]]) -> pd.Series:
    imp = pd.Series(np.abs(sv).mean(axis=0), index=cols)
    return pd.Series({g: imp[[c for c in cols if c in members]].sum() for g, members in groups.items()
                      if any(c in members for c in cols)})


def weather_effects(sv: np.ndarray, X: pd.DataFrame) -> pd.DataFrame:
    """기상 조건별 평균 SHAP 합(그 조건이 교통량을 평균 대비 몇 대 바꾸는지)."""
    weather_cols = [c for c in ["temperature_2m", "relative_humidity_2m", "precipitation", "rain", "snowfall",
                                "snow_depth", "cloud_cover", "wind_speed_10m", "weather_code", "is_rain", "is_snow"]
                    if c in X.columns]
    total = pd.Series(sv[:, [X.columns.get_loc(c) for c in weather_cols]].sum(axis=1), index=X.index)
    conds = {
        "눈 내림 (snowfall > 0)": X["snowfall"] > 0,
        "적설 5cm 이상": X["snow_depth"] >= 0.05,
        "비 (precip ≥ 0.1mm, 눈 아님)": (X["precipitation"] >= 0.1) & (X["snowfall"] == 0),
        "기온 −15°C 미만": X["temperature_2m"] < -15,
        "맑고 건조 (precip 0, 구름 < 20%)": (X["precipitation"] == 0) & (X["cloud_cover"] < 20),
    }
    rows = [{"조건": k, "시간 수": int(m.sum()), "기상 SHAP 합 평균(대/시간)": round(float(total[m].mean()), 1)}
            for k, m in conds.items() if m.sum() > 0]
    return pd.DataFrame(rows)


def run(cfg: dict, sample: int = 3000) -> None:
    import matplotlib.pyplot as plt

    out = Path(cfg["out"])
    shap_dir = out / "shap"
    shap_dir.mkdir(parents=True, exist_ok=True)
    groups = json.loads((out / "groups.json").read_text(encoding="utf-8"))
    test = pd.read_pickle(out / "models" / "test_features.pkl")
    Xs = test.sample(min(sample, len(test)), random_state=0)

    group_rows = {}
    for exp in ["P2", "P3"]:
        cols = json.loads((out / "models" / f"{exp}_features.json").read_text())
        model = joblib.load(out / "models" / f"{exp}_XGBoost.joblib")
        X = Xs[cols]
        sv = shap_values(model, X)
        imp = pd.Series(np.abs(sv).mean(axis=0), index=cols).sort_values(ascending=False)
        imp.round(2).to_csv(shap_dir / f"{exp}_mean_abs_shap.csv", header=["mean_abs_shap"])
        gi = group_importance(sv, cols, groups)
        group_rows[exp] = (gi / gi.sum() * 100).round(1)

        plt.figure()
        shap.summary_plot(sv, X, max_display=15, show=False, plot_size=(8, 6))
        plt.title(f"{exp} XGBoost SHAP (테스트 {len(X)}시간 표본)", loc="left")
        plots.save(plt.gcf(), shap_dir / f"{exp}_beeswarm.png")

        if exp == "P2":
            effects = weather_effects(sv, X)
            effects.to_csv(shap_dir / "P2_weather_effects.csv", index=False)
            (shap_dir / "P2_weather_effects.md").write_text(effects.to_markdown(index=False) + "\n", encoding="utf-8")
            print(effects.to_markdown(index=False))
            fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
            for ax, col, label in zip(axes, ["temperature_2m", "snowfall", "precipitation"],
                                      ["기온 (°C)", "강설 (cm/시간)", "강수 (mm/시간)"]):
                j = cols.index(col)
                ax.scatter(X[col], sv[:, j], s=8, alpha=0.4, color=plots.SERIES[2], edgecolors="none")
                ax.axhline(0, color=plots.INK_2, lw=0.8)
                ax.set_xlabel(label)
            axes[0].set_ylabel("SHAP (대/시간)")
            fig.suptitle("P2: 기상 변수 값에 따른 교통량 기여", x=0.01, ha="left", fontweight="bold")
            plots.save(fig, shap_dir / "P2_weather_dependence.png")

    table = pd.DataFrame(group_rows).T.fillna(0)
    table.to_csv(shap_dir / "group_share.csv")
    fig, ax = plt.subplots(figsize=(8, 2.6))
    left = np.zeros(len(table))
    for g in ["calendar", "weather", "lag"]:
        if g not in table:
            continue
        vals = table[g].to_numpy()
        ax.barh(table.index, vals, left=left, color=GROUP_COLOR[g], label=GROUP_LABEL[g],
                edgecolor=plots.SURFACE, linewidth=2, height=0.55)
        for y, (l, v) in enumerate(zip(left, vals)):
            if v >= 6:
                ax.text(l + v / 2, y, f"{v:.0f}%", ha="center", va="center", color="white", fontsize=10)
        left += vals
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.set_xlabel("평균 |SHAP| 비중 (%)")
    ax.set_title("요인 그룹별 기여도")
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=3, loc="lower right", bbox_to_anchor=(1, 1.0), fontsize=9)
    plots.save(fig, shap_dir / "group_share.png")
    print(table)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--sample", type=int, default=3000)
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        run(yaml.safe_load(f), args.sample)


if __name__ == "__main__":
    main()
