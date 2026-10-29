import numpy as np

from src.vision.counter import LineCounter, events_to_timeseries
from src.vision.tracker import IoUTracker, iou
from src.vision.types import Detection


def moving_box(step: int, x: float = 100, y0: float = 0, dy: float = 10) -> Detection:
    y = y0 + dy * step
    return Detection(bbox=(x, y, x + 50, y + 40), cls="car")


def test_iou_basic():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert abs(iou((0, 0, 10, 10), (5, 0, 15, 10)) - 1 / 3) < 1e-9


def test_tracker_keeps_one_id_for_smooth_motion():
    tracker = IoUTracker(iou_threshold=0.3, max_age=5, min_hits=2)
    ids = set()
    for step in range(20):
        tracks = tracker.update([moving_box(step)])
        ids |= {t.id for t in tracks if t.confirmed}
    assert ids == {1}


def test_tracker_survives_short_gap_with_velocity_prediction():
    tracker = IoUTracker(iou_threshold=0.3, max_age=5, min_hits=2)
    for step in range(10):
        tracker.update([moving_box(step)])
    for _ in range(3):  # 3프레임 동안 검출 누락
        tracker.update([])
    tracks = tracker.update([moving_box(13)])
    assert [t.id for t in tracks if t.confirmed] == [1]


def test_tracker_two_objects_get_two_ids():
    tracker = IoUTracker(iou_threshold=0.3, max_age=5, min_hits=2)
    for step in range(10):
        tracks = tracker.update([moving_box(step, x=100), moving_box(step, x=400)])
    assert sorted(t.id for t in tracks if t.confirmed) == [1, 2]


def test_counter_counts_each_track_once_with_direction_and_lane():
    tracker = IoUTracker(iou_threshold=0.3, max_age=5, min_hits=2)
    counter = LineCounter(
        p1=(0, 100),
        p2=(600, 100),
        directions={"positive": "down", "negative": "up"},
        lanes=[{"name": "L1", "t": [0.0, 0.5]}, {"name": "L2", "t": [0.5, 1.0]}],
        anchor="center",
    )
    events = []
    for step in range(30):  # 아래로 이동하며 y=100 선을 한 번 지남
        dets = [moving_box(step, x=100), moving_box(step, x=400)]
        events += counter.update(tracker.update(dets), frame_idx=step, fps=25.0)
    # 선 근처에서 앞뒤로 흔들려도 같은 트랙은 다시 세지 않는다
    for step in range(30, 0, -1):
        dets = [moving_box(step, x=100), moving_box(step, x=400)]
        events += counter.update(tracker.update(dets), frame_idx=60 - step, fps=25.0)
    assert len(events) == 2
    assert {e.direction for e in events} == {"down"}
    assert sorted(e.lane for e in events) == ["L1", "L2"]
    assert all(e.cls == "car" for e in events)


def test_counter_direction_sign():
    counter = LineCounter(p1=(0, 100), p2=(600, 100), directions={"positive": "down", "negative": "up"})
    tracker = IoUTracker(iou_threshold=0.3, max_age=5, min_hits=2)
    events = []
    for step in range(30):
        events += counter.update(
            tracker.update([moving_box(step, y0=300, dy=-10)]), frame_idx=step, fps=25.0
        )
    assert [e.direction for e in events] == ["up"]


def test_events_to_timeseries_bins_and_fills_zero():
    import pandas as pd

    events = pd.DataFrame(
        {
            "time": pd.to_datetime(
                ["2026-10-29 08:01", "2026-10-29 08:14", "2026-10-29 08:40"]
            ),
            "direction": ["down", "down", "up"],
            "cls": ["car", "truck", "car"],
        }
    )
    ts = events_to_timeseries(
        events, freq="15min", start="2026-10-29 08:00", end="2026-10-29 09:00"
    )
    assert list(ts["volume"]) == [2, 0, 1, 0]
    assert list(ts["down"]) == [2, 0, 0, 0]
    assert np.array_equal(ts.index, pd.date_range("2026-10-29 08:00", periods=4, freq="15min"))
