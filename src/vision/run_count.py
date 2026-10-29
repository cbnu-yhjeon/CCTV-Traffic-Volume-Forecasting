"""녹화 영상에 카운팅 실험(C1~C3)을 돌려 이벤트 CSV와 요약을 남긴다.

    python -m src.vision.run_count --config configs/sites/traffic_mp4.yaml --method C1 C2 C3
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import pandas as pd

from src.vision.counter import events_to_timeseries
from src.vision.pipeline import METHODS, CountingPipeline, draw, iter_frames, load_config


def run(cfg: dict, method: str, out_dir: Path, save_video: bool, snapshot_frames: list[int]) -> dict:
    cap = cv2.VideoCapture(cfg["source"])
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()

    pipe = CountingPipeline(cfg, method)
    writer = None
    if save_video:
        (out_dir / "videos").mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(out_dir / "videos" / f"{cfg['name']}_{method}.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"), fps, (size[0] // 2, size[1] // 2),
        )

    events = []
    t0 = time.perf_counter()
    n = 0
    for idx, frame in iter_frames(cfg["source"]):
        _, tracks, new = pipe.step(frame, idx, fps)
        events += new
        n += 1
        if writer is not None or idx in snapshot_frames:
            vis = draw(frame, cfg, tracks, len(events), method)
            if writer is not None:
                writer.write(cv2.resize(vis, (size[0] // 2, size[1] // 2)))
            if idx in snapshot_frames:
                cv2.imwrite(str(out_dir / f"{cfg['name']}_{method}_f{idx}.jpg"),
                            cv2.resize(vis, (size[0] // 2, size[1] // 2)))
    elapsed = time.perf_counter() - t0
    if writer is not None:
        writer.release()

    df = pd.DataFrame([e.to_dict() for e in events],
                      columns=["frame", "seconds", "track_id", "direction", "lane", "cls"])
    start = pd.Timestamp(cfg.get("start_time") or "2000-01-01")
    df["time"] = start + pd.to_timedelta(df["seconds"], unit="s")
    df.to_csv(out_dir / f"{cfg['name']}_{method}_events.csv", index=False)

    summary = {
        "site": cfg["name"],
        "method": method,
        "detector": METHODS[method][0],
        "tracker": METHODS[method][1],
        "frames": n,
        "seconds": round(elapsed, 2),
        "fps": round(n / elapsed, 1),
        "total": len(df),
        "by_lane": df["lane"].value_counts().sort_index().to_dict(),
        "by_class": df["cls"].value_counts().sort_index().to_dict(),
        "by_direction": df["direction"].value_counts().sort_index().to_dict(),
    }
    if len(df):
        ts = events_to_timeseries(df, freq="15min")
        ts.to_csv(out_dir / f"{cfg['name']}_{method}_15min.csv")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--method", nargs="+", default=list(METHODS), choices=list(METHODS))
    ap.add_argument("--out", default="results/counting")
    ap.add_argument("--save-video", action="store_true", help="주석 영상 저장(저장소에는 올리지 않음)")
    ap.add_argument("--snapshots", type=int, nargs="*", default=[750])
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for method in args.method:
        s = run(cfg, method, out_dir, args.save_video, args.snapshots)
        print(json.dumps(s, ensure_ascii=False))
        summaries.append(s)
    with open(out_dir / f"{cfg['name']}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
