"""가상선 통과 카운팅과 시계열 집계.

트랙의 기준점(중심 또는 하단 중앙)이 직전 프레임 → 현재 프레임 사이에 가상선 선분을
가로지르면 한 대로 센다. 트랙 ID마다 한 번만 센다. 방향은 선분 p1→p2에 대한 외적 부호,
차로는 교차점이 선분 위 어디(t ∈ [0, 1])에 있는지로 정한다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from src.vision.tracker import Track

Point = tuple[float, float]


@dataclass
class CountEvent:
    frame: int
    seconds: float
    track_id: int
    direction: str
    lane: str
    cls: str

    def to_dict(self) -> dict:
        return asdict(self)


def _cross(o: Point, a: Point, b: Point) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def segment_intersection(p: Point, q: Point, a: Point, b: Point) -> float | None:
    """선분 p→q가 선분 a→b와 만나면 a→b 위의 위치 t(0~1)를, 아니면 None."""
    r = (q[0] - p[0], q[1] - p[1])
    s = (b[0] - a[0], b[1] - a[1])
    denom = r[0] * s[1] - r[1] * s[0]
    if denom == 0:
        return None
    ap = (a[0] - p[0], a[1] - p[1])
    u = (ap[0] * r[1] - ap[1] * r[0]) / denom  # a→b 위의 위치
    v = (ap[0] * s[1] - ap[1] * s[0]) / denom  # p→q 위의 위치
    if 0 <= u <= 1 and 0 < v <= 1:
        return u
    return None


class LineCounter:
    def __init__(
        self,
        p1: Point,
        p2: Point,
        directions: dict[str, str] | None = None,
        lanes: list[dict] | None = None,
        anchor: str = "center",
    ) -> None:
        self.p1 = tuple(p1)
        self.p2 = tuple(p2)
        self.directions = directions or {"positive": "positive", "negative": "negative"}
        self.lanes = lanes or [{"name": "all", "t": [0.0, 1.0]}]
        self.anchor = anchor
        self._last: dict[int, Point] = {}
        self._counted: set[int] = set()

    def _anchor(self, track: Track) -> Point:
        x1, y1, x2, y2 = track.bbox
        if self.anchor == "bottom":
            return (x1 + x2) / 2, y2
        return (x1 + x2) / 2, (y1 + y2) / 2

    def _lane(self, t: float) -> str:
        for lane in self.lanes:
            lo, hi = lane["t"]
            if lo <= t <= hi:
                return lane["name"]
        return self.lanes[-1]["name"]

    def update(self, tracks: list[Track], frame_idx: int, fps: float) -> list[CountEvent]:
        events: list[CountEvent] = []
        for track in tracks:
            cur = self._anchor(track)
            prev = self._last.get(track.id)
            self._last[track.id] = cur
            if prev is None or track.id in self._counted or not track.confirmed:
                continue
            t = segment_intersection(prev, cur, self.p1, self.p2)
            if t is None:
                continue
            # 직전 위치의 외적 부호로 판정(현재 위치는 선 위에 정확히 걸칠 수 있다).
            # 음(선분 p1→p2의 한쪽) → 양으로 넘어가면 positive
            side = _cross(self.p1, self.p2, prev)
            direction = self.directions["positive" if side < 0 else "negative"]
            self._counted.add(track.id)
            events.append(
                CountEvent(
                    frame=frame_idx,
                    seconds=frame_idx / fps,
                    track_id=track.id,
                    direction=direction,
                    lane=self._lane(t),
                    cls=track.cls,
                )
            )
        return events


def events_to_timeseries(
    events: pd.DataFrame,
    freq: str = "15min",
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """카운트 이벤트(time, direction, cls)를 고정 간격 교통량 시계열로 집계.

    반환: index=구간 시작 시각, 열 = volume(전체) + 방향별 + 차종별(cls_*) 대수.
    이벤트가 없는 구간은 0으로 채운다(구간 [start, end)).
    """
    ev = events.copy()
    ev["bin"] = pd.to_datetime(ev["time"]).dt.floor(freq)
    start = pd.Timestamp(start) if start is not None else ev["bin"].min()
    end = pd.Timestamp(end) if end is not None else ev["bin"].max() + pd.tseries.frequencies.to_offset(freq)
    index = pd.date_range(start, end, freq=freq, inclusive="left")

    out = pd.DataFrame(index=index)
    out.index.name = "timestamp"
    out["volume"] = ev.groupby("bin").size().reindex(index, fill_value=0)
    for direction, grp in ev.groupby("direction"):
        out[direction] = grp.groupby("bin").size().reindex(index, fill_value=0)
    for cls, grp in ev.groupby("cls"):
        out[f"cls_{cls}"] = grp.groupby("bin").size().reindex(index, fill_value=0)
    return out.astype(int)
