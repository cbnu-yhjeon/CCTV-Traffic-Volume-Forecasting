"""수작업 카운트 보조 도구: 가상선 위 픽셀을 프레임마다 한 줄씩 쌓은 슬릿스캔 이미지.

가로축 = 가상선 위치(차로), 세로축 = 시간. 선을 지나는 차량 한 대가 덩어리 하나로
보이므로, 영상을 처음부터 끝까지 보지 않아도 통과 대수와 시각을 셀 수 있다.

    python -m src.eval.slitscan --config configs/sites/traffic_mp4.yaml --out results/counting/slitscan
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.vision.pipeline import iter_frames, load_config


def line_pixels(p1, p2, step: int = 2) -> tuple[np.ndarray, np.ndarray]:
    n = int(np.hypot(p2[0] - p1[0], p2[1] - p1[1]) // step)
    xs = np.linspace(p1[0], p2[0], n).round().astype(int)
    ys = np.linspace(p1[1], p2[1], n).round().astype(int)
    return xs, ys


def build_slitscan(cfg: dict, step: int = 2, band: int = 1) -> tuple[np.ndarray, float]:
    """(프레임 수, 선 길이/step, 3) 이미지. band>0이면 선 위아래 ±band 픽셀 평균."""
    xs, ys = line_pixels(cfg["line"]["p1"], cfg["line"]["p2"], step)
    cap = cv2.VideoCapture(cfg["source"])
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    rows = []
    for _, frame in iter_frames(cfg["source"]):
        strip = np.stack([frame[ys + d, xs] for d in range(-band, band + 1)]).mean(axis=0)
        rows.append(strip.astype(np.uint8))
    return np.stack(rows), fps


def render_chunks(scan: np.ndarray, fps: float, cfg: dict, out: Path, seconds: int = 10,
                  stretch: int = 2) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    per = int(seconds * fps)
    width = scan.shape[1]
    paths = []
    for k, start in enumerate(range(0, len(scan), per)):
        chunk = scan[start:start + per]
        img = cv2.resize(chunk, (width, len(chunk) * stretch), interpolation=cv2.INTER_NEAREST)
        canvas = np.full((img.shape[0], width + 70, 3), 255, np.uint8)
        canvas[:, 70:] = img
        for lane in cfg.get("lanes", [])[1:]:  # 차로 경계
            x = 70 + int(lane["t"][0] * width)
            cv2.line(canvas, (x, 0), (x, img.shape[0]), (0, 0, 255), 1)
        half = int(np.ceil(start / fps * 2))
        while (f := round(half * fps / 2)) < start + len(chunk):  # 0.5초마다 눈금
            y = (f - start) * stretch
            full = half % 2 == 0
            cv2.line(canvas, (40 if full else 58, y), (70, y), (0, 0, 0), 1)
            if full:
                cv2.putText(canvas, f"{half // 2}s", (2, y + 14), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (0, 0, 0), 1)
            half += 1
        path = out / f"slitscan_{k:02d}_{start / fps:04.0f}s.png"
        cv2.imwrite(str(path), canvas)
        paths.append(path)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="results/counting/slitscan")
    ap.add_argument("--seconds", type=int, default=10)
    args = ap.parse_args()
    cfg = load_config(args.config)
    scan, fps = build_slitscan(cfg)
    for p in render_chunks(scan, fps, cfg, Path(args.out), seconds=args.seconds):
        print(p)


if __name__ == "__main__":
    main()
