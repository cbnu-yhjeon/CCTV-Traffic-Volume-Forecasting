from pathlib import Path

import numpy as np
import pandas as pd

from src.collect.its_cctv import nearest, parse_response
from src.collect.run_collector import BinAccumulator, ViewMonitor

FIXTURE = Path(__file__).parent / "fixtures" / "its_cctv_sample.xml"


def test_parse_its_xml():
    cctvs = parse_response(FIXTURE.read_text(encoding="utf-8"))
    assert len(cctvs) == 2
    assert cctvs[0].name == "[수도권제1순환선] 성남"
    assert cctvs[0].format == "HLS"
    assert abs(cctvs[0].lat - 37.42889) < 1e-6 and abs(cctvs[0].lon - 127.12361) < 1e-6


def test_parse_its_json():
    text = '{"response": {"data": [{"cctvname": "A", "cctvurl": "http://x", "coordx": 127.0, "coordy": 36.0}]}}'
    (c,) = parse_response(text)
    assert (c.name, c.url, c.lat, c.lon) == ("A", "http://x", 36.0, 127.0)


def test_nearest_picks_closest():
    cctvs = parse_response(FIXTURE.read_text(encoding="utf-8"))
    assert nearest(cctvs, 37.4391, 127.1226).name == "[수도권제1순환선] 성남영업소"


def test_view_monitor_detects_scene_change_not_noise():
    rng = np.random.default_rng(0)
    scene = rng.integers(0, 255, (360, 640, 3), dtype=np.uint8)
    scene = np.ascontiguousarray(np.repeat(np.repeat(scene[::40, ::40], 40, 0), 40, 1)[:360, :640])
    mon = ViewMonitor(threshold=0.6)
    assert not mon.check(scene)
    noisy = np.clip(scene.astype(int) + rng.integers(-10, 10, scene.shape), 0, 255).astype(np.uint8)
    assert not mon.check(noisy)
    other = np.ascontiguousarray(np.flipud(np.fliplr(scene)))
    assert mon.check(other)


def test_bin_accumulator_writes_coverage_and_counts(tmp_path):
    acc = BinAccumulator(tmp_path / "counts.csv", "15min")
    t0 = pd.Timestamp("2026-10-29 08:00")
    for s in range(0, 15 * 60, 1):  # 08:00~08:15 전부 관측
        acc.observe(t0 + pd.Timedelta(seconds=s), 1.0)
    for s in range(450):  # 08:15~ 절반만 관측
        acc.observe(t0 + pd.Timedelta(minutes=15, seconds=s), 1.0)
    acc.add_event(t0 + pd.Timedelta(minutes=3), "north", "car")
    acc.add_event(t0 + pd.Timedelta(minutes=16), "south", "truck")
    acc.mark_view_change(t0 + pd.Timedelta(minutes=20))

    acc.flush(before=t0 + pd.Timedelta(minutes=15))  # 첫 구간만 닫힘
    first = pd.read_csv(tmp_path / "counts.csv")
    assert len(first) == 1 and first.loc[0, "volume"] == 1 and first.loc[0, "coverage"] == 1.0

    df = acc.flush()
    assert list(df["volume"]) == [1, 1]
    assert list(df["coverage"]) == [1.0, 0.5]
    assert list(df["view_change"]) == [0, 1]
    assert df.loc[1, "cls_truck"] == 1 and df.loc[0, "north"] == 1
