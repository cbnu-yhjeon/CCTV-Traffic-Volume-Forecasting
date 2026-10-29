# 수작업 카운트

| 파일 | 영상 | 작성 방법 |
|---|---|---|
| `traffic_mp4_manual.csv` | `traffic.mp4` (60초, 주간 · 맑음) | 가상선 슬릿스캔(`python -m src.eval.slitscan`)에서 차량 덩어리를 세고, 애매한 구간(0초, 22~24초, 41~44초, 59~60초)은 원본 프레임으로 확인 |

- `seconds`: 차량 중심이 가상선을 지난 시각(슬릿스캔 점유 구간의 중앙, 오차 약 ±0.3초)
- `cls`: car / motorcycle / van. 평가에서는 small(car, motorcycle) · large(van, truck, bus) 두 그룹으로 묶어 비교
- 0~1초에 지난 2대는 MOG2 배경 모델 워밍업(25프레임) 구간이라 평가 창(`--start 1.0`)에서 빠짐
- Claude가 작성한 카운트입니다. 슬릿스캔 이미지(`results/counting/slitscan/`)로 사람이 다시 검수할 수 있습니다.
