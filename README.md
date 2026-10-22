# 🐹 HamsterSuper — 햄스터봇 자율주행 (심층 강화학습)

Robomation **햄스터봇**이 미로형 레이스 코스를 **출발 칸 → 도착 칸**까지 최대한 빠르고 충돌 없이 달리도록
만드는 프로젝트입니다. 시뮬레이터에서 **PPO 강화학습**으로 학습하고, 실제 로봇에서는 **카메라 + Python 서버**로
같은 정책을 그대로 실행합니다. 학습 중 **500 iteration마다 GitHub Release**로 모델이 자동 배포됩니다.

## 코스 규격 (실측치 그대로 사용)

| 항목 | 값 |
|---|---|
| 바닥 점 간격 | 9 cm 격자 |
| 벽 | 가로 9 cm × 세로(두께) 0.5 cm × 높이 2.5 cm |
| 벽 배치 | 인접한 두 점을 잇는 위치에만 (가로/세로) |
| 통로 폭 | 9 − 0.5 = 8.5 cm (햄스터 폭 ≈ 3.5 cm) |

## 구조

```
카메라 ─▶ 벽 분할 ─▶ 지면기하 거리측정(21개 광선) ─┐
IR 근접센서(좌/우) ───────────────────────────────┤
                                                 ├─▶ PPO 정책(MLP) ─▶ 바퀴 속도(좌/우)
지도작성(격자 스냅) ─▶ BFS 경로계획 ─▶ 다음 경유점 ─┤
추측항법 + 격자 스냅 위치보정 ─────────────────────┘
```

* **시뮬레이터** (`hamster_super/sim`): 차동구동 동역학, 벽 충돌(원-사각형), 카메라 광선, IR 센서, 1인칭 카메라 렌더러.
* **도메인 랜덤화**: 바퀴 게인 ±15 %, 최고속도 14–26 cm/s, 모터 지연, 카메라/통신 지연 0–2 스텝, 바퀴 미끄러짐,
  센서 노이즈·누락·오검출, 위치추정 오차까지 매 에피소드 무작위 → 실제 환경의 "역경"에 강인하도록.
* **커리큘럼**: 2×1 직선 통로 → 최대 12×12 미로까지, 성공률 80 % 넘으면 자동 승급. 미로 / 순환미로 / 방 / 흩어진 벽 4가지 스타일.
* **Sim-to-Real 핵심**: 정책은 픽셀이 아니라 **카메라에서 계산한 21개 거리값**을 봅니다. 카메라 높이·각도를 알면 벽 아랫변의
  이미지 행이 곧 거리이므로(지면 기하), 시뮬레이션과 실제가 **동일한 입력 형식**을 갖습니다.
  카메라가 벽에 너무 가까우면 벽 너머를 보는 현상(사각지대)까지 시뮬레이션하여 IR 센서를 함께 쓰도록 학습합니다.
* **지도/위치추정**: 벽은 반드시 9 cm 격자선 위에 있다는 사실을 이용 — 관측된 벽 점을 격자 변에 스냅하여 지도를 만들고,
  같은 원리로 추측항법 오차를 보정합니다 (햄스터는 엔코더가 없음).

## 설치

```bash
pip install -r requirements.txt        # 실제 로봇은 roboid 포함
python -m pytest                        # 테스트
```

## 학습 (+ 500회마다 자동 릴리스)

```bash
gh auth login            # 또는 export GITHUB_TOKEN=...  (repo 권한)
python -m hamster_super.rl.train --iters 5000 --envs 32 --workers 4 --out runs/ppo
```

* 1 iteration = 32 env × 128 step = 4,096 스텝의 PPO 업데이트.
* **500 iteration마다** (`--release-every 500`):
  체크포인트 저장 → TorchScript/NumPy 내보내기 → 무작위 최고난도 맵 + `maps/` 예제맵 평가 →
  GitHub Release `model-iter000500`, `model-iter001000`, … 생성 (모델 파일·평가지표 첨부).
* 에피소드 단위로 원하면 `--release-unit episode`. 업로드 없이 확인만: `--release-dry-run`.
* 이어서 학습: `--resume runs/ppo/last.pt`. 진행 로그: `runs/ppo/train_log.jsonl`.

평가:
```bash
python -m hamster_super.rl.evaluate --model models/policy.npz --level 5 --episodes 100
python -m hamster_super.rl.evaluate --model models/policy.npz --maps maps/ --full-stack   # 추측항법+위치보정까지
```

## 실제 로봇 실행 (Python 서버)

1. 햄스터 위에 카메라 장착 (기본값: 바닥에서 렌즈 5 cm, 중심에서 앞으로 1.5 cm, 아래로 32° 기울임, 수평화각 70°).
   USB 웹캠 또는 ESP32-CAM 같은 Wi-Fi 스트림 모두 가능 (`--camera 0` 또는 `--camera http://IP:81/stream`).
2. 보정:
   ```bash
   python -m hamster_super.robot.calibrate speed     # 직진 속도
   python -m hamster_super.robot.calibrate turn      # 바퀴 간격
   python -m hamster_super.robot.calibrate camera    # 벽 인식 확인 (calib_mask.png)
   ```
   `calib.json` 예시:
   ```json
   {"robot": {"max_speed": 18.5, "track_width": 3.1},
    "camera": {"mount_height": 5.0, "pitch_deg": 32, "hfov_deg": 70},
    "segmentation": {"method": "dark", "dark_threshold": 110}}
   ```
3. 서버 실행 후 브라우저에서 `http://localhost:8000` (실시간 지도·경로·카메라·바퀴 상태, 시작/정지 버튼):
   ```bash
   # 코스 지도를 모를 때 (주행하며 지도 작성)
   python -m hamster_super.server --model models/policy.npz --camera 0 --calib calib.json \
       --start 0,0,E --goal 5,3 --arena 6,4
   # 코스 지도를 알 때
   python -m hamster_super.server --model models/policy.npz --map maps/race.txt --calib calib.json
   ```
4. 하드웨어 없이 디지털 트윈으로 전체 파이프라인(렌더링 → 인식 → 지도 → 정책) 확인:
   ```bash
   python -m hamster_super.server --model models/policy.npz --sim maps/example_02_maze.txt
   ```

## 맵 파일 형식

`maps/*.txt` — `+`는 바닥 점, `---`/`|`는 벽, `S`(+방향 E/N/W/S)는 출발, `G`는 도착:

```
+---+---+---+---+
| SE        |   |
+---+---+   +   +
|           |   |
+   +---+---+   +
|             G |
+---+---+---+---+
```

JSON도 지원: `{"walls": [[[x1,y1],[x2,y2]], ...], "start": [0,2], "goal": [3,0], "start_heading": "E"}` (점 좌표 = 9 cm 단위).

## 디렉터리

| 경로 | 내용 |
|---|---|
| `hamster_super/maps` | 격자 맵, 파서, 랜덤 생성기 |
| `hamster_super/sim` | 물리·센서·카메라 모델/렌더러 |
| `hamster_super/nav` | 지도작성, 경로계획, 위치추정, 관측 생성 |
| `hamster_super/env` | Gymnasium 환경, 멀티프로세스 벡터 환경 |
| `hamster_super/rl` | PPO 학습, 평가, 정책 내보내기 |
| `hamster_super/release` | GitHub Release 자동 업로드 |
| `hamster_super/robot` | 햄스터(roboid) 드라이버, 카메라, 인식, 제어 루프, 보정 |
| `hamster_super/server` | HTTP 서버 + 대시보드 |
| `models/` | 사전학습 정책 |

## 다음 단계

* **코스 사진이 오면**: 벽/바닥 색에 맞게 `segmentation` 조정 (필요시 작은 CNN 분할기로 교체 — 렌더러로 합성 데이터 생성 가능),
  사진 → 맵 자동 변환기 추가.
* 실제 주행 로그로 `max_speed`, 지연, 노이즈 범위를 재보정하고 그 범위로 재학습.
