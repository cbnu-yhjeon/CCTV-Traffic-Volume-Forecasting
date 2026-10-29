"""국가교통정보센터(ITS) CCTV 화상정보 API.

응답 하나에 CCTV 여러 대의 이름 · 좌표 · 스트림 URL(HLS)이 들어 있다. 스트림 URL에는 만료되는
토큰이 붙어 있으므로, 수집기는 주기적으로 목록을 다시 받아 URL을 갱신해야 한다.

- 인증키: https://www.its.go.kr/opendata/ 에서 신청 → 환경 변수 ITS_API_KEY
- apiKey=test 는 고정 샘플 목록만 주고 스트림은 401로 막혀 있다(파서 시험용)

    python -m src.collect.its_cctv --bbox 127.3 36.5 127.6 36.8 --near 36.63 127.49 --snapshot site.jpg
"""

from __future__ import annotations

import argparse
import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import requests

API_URL = "https://openapi.its.go.kr:9443/cctvInfo"


@dataclass
class CctvInfo:
    name: str
    url: str
    lon: float
    lat: float
    format: str = ""
    road_section_id: str = ""


def parse_response(text: str) -> list[CctvInfo]:
    """XML · JSON 응답 모두 처리한다(키 · 서버 설정에 따라 getType이 무시되기도 한다)."""
    text = text.strip()
    if text.startswith("{"):
        import json

        payload = json.loads(text)
        items = payload.get("response", payload).get("data", [])
        if isinstance(items, dict):
            items = [items]
        rows = [{k.lower(): v for k, v in item.items()} for item in items]
    else:
        root = ET.fromstring(text)
        rows = [{child.tag.lower(): (child.text or "") for child in data} for data in root.iter("data")]
    return [
        CctvInfo(
            name=r.get("cctvname", "").strip(),
            url=r.get("cctvurl", "").strip(),
            lon=float(r.get("coordx") or "nan"),
            lat=float(r.get("coordy") or "nan"),
            format=r.get("cctvformat", "") or "",
            road_section_id=r.get("roadsectionid", "") or "",
        )
        for r in rows
    ]


def list_cctvs(
    min_x: float, min_y: float, max_x: float, max_y: float,
    api_key: str | None = None, road_type: str = "its", cctv_type: int = 1,
) -> list[CctvInfo]:
    """road_type: its(국도) | ex(고속도로). cctv_type: 1=실시간 스트리밍(HLS), 2=동영상 파일."""
    params = {
        "apiKey": api_key or os.environ.get("ITS_API_KEY", "test"),
        "type": road_type, "cctvType": cctv_type,
        "minX": min_x, "maxX": max_x, "minY": min_y, "maxY": max_y,
        "getType": "json",
    }
    resp = requests.get(API_URL, params=params, timeout=30)
    resp.raise_for_status()
    return parse_response(resp.text)


def nearest(cctvs: list[CctvInfo], lat: float, lon: float) -> CctvInfo:
    def dist(c: CctvInfo) -> float:
        dx = (c.lon - lon) * math.cos(math.radians(lat))
        return math.hypot(dx, c.lat - lat)

    return min(cctvs, key=dist)


def find_by_name(cctvs: list[CctvInfo], name: str) -> CctvInfo | None:
    return next((c for c in cctvs if name in c.name), None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", type=float, nargs=4, metavar=("MIN_X", "MIN_Y", "MAX_X", "MAX_Y"),
                    required=True, help="경도 · 위도 범위")
    ap.add_argument("--road-type", default="its", choices=["its", "ex"])
    ap.add_argument("--near", type=float, nargs=2, metavar=("LAT", "LON"))
    ap.add_argument("--snapshot", help="가장 가까운 CCTV의 한 프레임을 저장(가상선 좌표 잡기용)")
    args = ap.parse_args()
    cctvs = list_cctvs(*args.bbox, road_type=args.road_type)
    for c in cctvs:
        print(f"{c.name}\t{c.lat:.5f}\t{c.lon:.5f}\t{c.format}")
    if args.near and cctvs:
        cctv = nearest(cctvs, *args.near)
        print("nearest:", cctv)
        if args.snapshot:
            import cv2

            ok, frame = cv2.VideoCapture(cctv.url).read()
            if not ok:
                raise SystemExit("stream read failed (인증키 · 토큰 만료 확인)")
            cv2.imwrite(args.snapshot, frame)
            print("saved", args.snapshot, frame.shape)


if __name__ == "__main__":
    main()
