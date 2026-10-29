from src.vision.detect_mog2 import merge_fragments
from src.vision.types import Detection, suppress_contained


def test_merge_fragments_joins_vertically_split_car():
    windshield = (100, 100, 300, 150)
    body = (95, 170, 305, 300)  # 세로 간격 20
    assert merge_fragments([windshield, body], max_gap=40) == [(95, 100, 305, 300)]


def test_merge_fragments_keeps_side_by_side_cars_apart():
    left = (100, 100, 300, 300)
    right = (350, 110, 550, 310)  # 가로로 안 겹침
    assert len(merge_fragments([left, right], max_gap=40)) == 2


def test_merge_fragments_keeps_far_apart_cars_in_same_lane():
    lead = (100, 400, 300, 600)
    follower = (110, 200, 290, 330)  # 세로 간격 70
    assert len(merge_fragments([lead, follower], max_gap=40)) == 2


def test_suppress_contained_drops_nested_box():
    van = Detection((100, 100, 400, 400), 0.8, "truck")
    roof = Detection((150, 100, 350, 180), 0.5, "truck")
    other = Detection((500, 100, 700, 300), 0.9, "car")
    kept = suppress_contained([roof, van, other])
    assert sorted(d.bbox for d in kept) == [van.bbox, other.bbox]
