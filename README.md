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
추측항법 + 격자 파티클필터 위치추정 ────────────────┘
```

* **시뮬레이터** (`hamster_super/sim`): 차동구동 동역학, 벽 충돌(원-사각형), 카메라 광선, IR 센서, 1인칭 카메라 렌더러.
* **도메인 랜덤화**: 바퀴 게인 ±15 %, 최고속도 14–26 cm/s, 모터 지연, 카메라/통신 지연 0–2 스텝, 바퀴 미끄러짐,
  센서 노이즈·누락·오검출, 위치추정 오차까지 매 에피소드 무작위 → 실제 환경의 "역경"에 강인하도록.
* **커리큘럼**: 2×1 직선 통로 → 최대 12×12 미로까지, 성공률 80 % 넘으면 자동 승급. 미로 / 순환미로 / 방 / 흩어진 벽 4가지 스타일.
* **Sim-to-Real 핵심**: 정책은 픽셀이 아니라 **카메라에서 계산한 21개 거리값**을 봅니다. 카메라 높이·각도를 알면 벽 아랫변의
  이미지 행이 곧 거리이므로(지면 기하), 시뮬레이션과 실제가 **동일한 입력 형식**을 갖습니다.
  카메라가 벽에 너무 가까우면 벽 너머를 보는 현상(사각지대)까지 시뮬레이션하여 IR 센서를 함께 쓰도록 학습합니다.
* **지도/위치추정**: 벽은 반드시 9 cm 격자선 위에 있다는 사실을 이용 — 관측된 벽 점(카메라+IR)을 격자 변에 스냅하여 지도를 만들고,
  **파티클 필터(MCL)** 가 "벽 점은 격자선 위, 그리고 이미 그린 지도와 일치해야 한다"는 우도로 위치를 추정합니다.
  햄스터는 엔코더가 없어 명령 속도로 추측항법을 하는데, 각 파티클이 자신만의 속도 배율을 가져 실제 속도를 스스로 추정하고,
  벽에 걸려 헛도는 상황도 견딥니다.
* **현실형 학습**: 에피소드의 70 %는 실제 로봇과 똑같이 (보정 오차가 있는) 추측항법 + 파티클 필터로 주행하며 학습하므로,
  정책이 위치추정 오차 패턴에 적응합니다 (`--full-stack-prob`).

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

### GitHub Pages 학습 대시보드

👉 **https://menonng.github.io/hamsterSuper/** — 어디서나 브라우저로 학습 현황 확인.

```bash
python -m hamster_super.rl.train --iters 5000 --out runs/ppo --pages-every 50   # 50 iteration마다 Pages 갱신
python -m hamster_super.rl.pages --run runs/ppo                                 # 수동으로 한 번 게시
```

* 50 iteration마다 학습 곡선·릴리스 결과·최신 모델의 주행 녹화 3판을 JSON으로 만들어 `gh-pages` 브랜치에 푸시
  (매번 단일 커밋으로 덮어써서 저장소 용량이 늘지 않음). 페이지는 1분마다 데이터를 새로 읽고, 녹화를 애니메이션으로 재생합니다.
* 녹화·푸시는 백그라운드 스레드라 학습을 멈추지 않습니다. 푸시에는 `git push` 권한(로그인된 git)이 필요합니다.
* 배포는 GitHub Actions가 자동으로: `gh-pages` 푸시 → `Pages snapshot` 워크플로 → 기본 브랜치의 `Pages` 워크플로가
  배포 (Pages 소스가 "GitHub Actions"가 아니면 스스로 전환). 별도 설정 불필요.

### 실시간 학습 모니터 (HTML)

```bash
python -m hamster_super.rl.train --iters 5000 --out runs/ppo --monitor     # 학습 + 대시보드 함께 실행
python -m hamster_super.rl.monitor --run runs/ppo --port 8001              # 또는 이미 돌고 있는 학습에 따로 붙이기
```

브라우저에서 `http://localhost:8001` (같은 네트워크면 `http://<PC IP>:8001`):

* **학습 곡선**: 성공률, 에피소드 보상, 완주 시간, 벽 접촉, 커리큘럼 단계, 탐험(std) — 3초마다 자동 갱신,
  마우스를 올리면 해당 iteration 값 표시, 노란 점선은 500회 릴리스 지점.
* **3D 리플레이** (Three.js): 실제 크기(벽 9×0.5×2.5 cm)의 3D 미로에서 주행을 봅니다. 시점: 자유 회전 / 추적 카메라 /
  **로봇 카메라**(실제 카메라와 같은 높이 5 cm·하향 32°·화각 70° — 로봇이 실제로 보게 될 화면). 2D 보기로도 전환 가능.
* **라이브 리플레이**: 가장 최신 체크포인트가 무작위 미로를 실시간으로 주행 — 실제 벽 / 로봇이 찾은 벽,
  계획 경로, 실제 궤적과 로봇의 추정 위치, 카메라 광선이 그대로 보입니다. 체크포인트가 갱신되면(10 iteration마다) 다음 판부터 새 모델로 교체.
* **릴리스 표**: 각 릴리스의 평가 성공률·충돌·완주시간.
* 학습과 별도 프로세스라 학습 속도에 영향 없음. 옵션: `--level 3`(리플레이 미로 크기 고정), `--speed 2`(2배속), `--ideal-pose`.

* 1 iteration = 32 env × 128 step = 4,096 스텝의 PPO 업데이트.
* **500 iteration마다** (`--release-every 500`):
  체크포인트 저장 → TorchScript/NumPy 내보내기 → 무작위 최고난도 맵 + `maps/` 예제맵 평가 →
  GitHub Release `model-iter000500`, `model-iter001000`, … 생성 (모델 파일·평가지표 첨부).
* 릴리스 방법 (`--release-method auto`가 자동 선택):
  1. `gh` CLI 로그인되어 있으면 → `gh release create`
  2. `GITHUB_TOKEN` 환경변수가 있으면 → GitHub API
  3. 둘 다 없으면 → **브랜치 푸시만으로**: 모델 파일을 `model-releases` 브랜치에 푸시하면
     `.github/workflows/model-release.yml` 워크플로가 저장소 기본 토큰으로 릴리스를 만들고 파일을 첨부합니다.
     토큰 설정 불필요, 태그 푸시 권한이 없는 환경(예: Claude 클라우드 세션)에서도 동작.
* 이미 저장된 릴리스 폴더를 나중에 올리기:
  ```bash
  python -m hamster_super.release.github_release releases/model-iter*
  ```
* 에피소드 단위로 원하면 `--release-unit episode`. 업로드 없이 확인만: `--release-dry-run`.
* 이어서 학습: `--resume runs/ppo/last.pt`. 진행 로그: `runs/ppo/train_log.jsonl`.

평가:
```bash
python -m hamster_super.rl.evaluate --model models/policy.npz --level 5 --episodes 100
python -m hamster_super.rl.evaluate --model models/policy.npz --maps maps/ --full-stack   # 추측항법+위치보정까지
```

## 실제 로봇 실행 (Python 서버)

1. 카메라 장착: 햄스터와 같은 크기의 카메라 모듈을 **높이 1.5 cm 부착물**로 햄스터 위에 연결, 진행 방향을 향하게.
   기본값: 렌즈 높이 6.0 cm(햄스터 3.0 + 부착물 1.5 + 모듈 절반 1.5), 중심에서 앞으로 2.0 cm, 수평화각 70°,
   아래로 35° 기울임(각도는 자유지만 한 번 맞추면 고정 → 아래 `pitch` 보정으로 실제 각도를 측정).
   코스: **바닥 검정, 벽 하양** → 벽 인식은 밝기 기준(프레임마다 자동 임계값, `otsu`).
   USB 웹캠 또는 ESP32-CAM 같은 Wi-Fi 스트림 모두 가능 (`--camera 0` 또는 `--camera http://IP:81/stream`).
2. 보정:
   ```bash
   python -m hamster_super.robot.calibrate speed     # 직진 속도 (추측항법 정확도에 중요!)
   python -m hamster_super.robot.calibrate turn      # 바퀴 간격
   python -m hamster_super.robot.calibrate pitch --distance 10   # 벽을 렌즈 앞 10 cm에 두고 카메라 기울기 측정
   python -m hamster_super.robot.calibrate camera    # 벽 인식 확인 (calib_mask.png)
   ```
   `calib.json` 예시:
   ```json
   {"robot": {"max_speed": 18.5, "track_width": 3.1},
    "camera": {"mount_height": 6.0, "pitch_deg": 35.4, "hfov_deg": 70},
    "segmentation": {"method": "otsu"}}
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
