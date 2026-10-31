# B E7 작업 기록 (두 번째 공개 DINO-WM 과제: PointMaze, RunPod)

- 작업자: B E7 담당(원고 세션 `near-miss-blindness-tmlr`이 시작한 세션). 시각은 UTC.
- 기준: `main/DESIGN_main.md` v1~§11(현재 sha256 ea176a5c…), [reports/E1_server.md](../reports/E1_server.md), E1 코드(`main/code/server/e1lib.py`, `e1_calib.py`, `e1_judgeset.py`, `e1_judges.py`, `e1_m1_eval.py`, `e1_m1_analyze.py`).
- **상태: 사용자 승인(2026-10-31T11:13Z, "승인, 진행").** 아래 A절 문안을 그대로 `DESIGN_main.md` §12로 덧붙였고 sha256과 시각을 LOG.md에 적었다. 그 뒤에 pod를 만든다.

## A. DESIGN_main.md §12 문안 (11:04Z 초안 → 11:13Z 승인; DESIGN_main.md §12와 같은 문안)

## 12. E7 덧붙임 (2026-10-31T11:13Z, 사용자 승인; 평가 집합을 상상하기 전): 두 번째 공개 DINO-WM 과제(PointMaze)

### 12.0 목적과 범위
- M1(공개 DINO-WM PushT)의 경계 해상도 결과(H1b 성립, 양방향 흐림, horizon 의존, 상상 특징 probe 차 H4b)가 한 과제·한 체크포인트에 그친다는 약점을 보완한다. 같은 모델 계열(DINO-WM)의 **두 번째 공개 체크포인트·과제(PointMaze)**에서 E1 M1 프로토콜을 그대로 반복한다.
- 1~11절의 판정과 집계 규칙(v2.1 "모델 2개 이상")은 바꾸지 않는다. E7은 모델 수를 늘리는 것으로 세지 않고(M1과 같은 모델 계열), **사전등록된 반복(replication) 단계**로 따로 판정·보고한다.
- 후보였던 Wall(`wall_single`)은 쓰지 않는다. 이유(2026-10-31 CPU 확인): 공개 환경 코드의 행동→이동 규약(`DotWall._generate_transition`: `location + action * 2`)이 학습 데이터(`s[t+1] = s[t] + a[t]`, 벽 접촉 없는 38,883 step에서 잔차 1e-6, 정확히 1배)와 2배 다르다. 데이터의 벽 접촉 처리도 결정론적 환경 코드와 달라 1배로 고쳐도 궤적 1,920개 중 14%만 끝까지 재생된다(기록 step의 30%가 접촉). 궤적 길이 50이라 H = 10도 불가능하다. 이 불일치는 A 논문의 Wall 계획 결과 해석에도 관련되므로 사용자에게 따로 알린다.

### 12.1 실행 장소와 예산
- **RunPod**에서 E7 전체(보정·판별기·곡선·분석)를 pod 하나에서 돈다. 서버 GPU 두 장이 사용자의 다른 작업(`imlewm.train`, tmux e16/q0/q1)으로 점유 중이라 사용자가 장소를 RunPod로 정했다(2026-10-31). 0.1절 "한 비교 안에서 장소를 섞지 않는다"는 E7 안에서 지켜진다(장소 하나).
- GPU: NVIDIA A40(secure, $0.59/h) 우선, 없으면 RTX A6000 → RTX 4090. 이미지: runpod-torch-v240(torch 2.4.1+cu124, Python 3.11, Ubuntu 22.04; E1 서버 환경 b-dinowm과 같은 torch 판). pod 자동 종료 6시간(TERM_H 6).
- 예산: 사용자 상한 **$30**. 도구는 `runpod_main` 사본 `B_near_miss_blindness/workspace/runpod_e7/`(pod 접두어 `bx7`, 자체 장부 `pods.ledger`, START_BALANCE는 pod 생성 직전 잔액). 감시 `budget.py watch 300`은 LIMIT **$28**에서 pod를 지우고, 단계 cap은 `caps.json {"e7": 20}`이다. 추정 비용은 12.9절.
- E1 M1(서버 A5000)과 E7(RunPod)은 장소가 다르다. 두 과제의 수치를 합치지 않고 과제별로 따로 보고한다.

### 12.2 모델·과제·데이터
| 항목 | 내용 |
|---|---|
| 월드모델 | DINO-WM 공개 PointMaze 체크포인트 `outputs/point_maze/checkpoints/model_latest.pth`(OSF `outputs.zip`, sha256 `78c654baede0e9289b751e9ad251839b8045a6074cc1ec23ab54caaf11a25700`), `hydra.yaml`(sha256 `25acae83…28bcda`): DINOv2 ViT-S/14 인코더(고정), ViT 예측기 depth 6, num_hist 3, frameskip 5, img_size 224(인코더 입력 196 px, 패치 14 × 14 = 196). PushT 체크포인트와 같은 구조·설정이다 |
| 환경 | DINO-WM `point_maze`(U_MAZE 5 × 5 칸, mujoco-py 2.1.2.14 + d4rl@89141a6, dino_wm commit 0a9492f). 상태 (x, y, vx, vy), 행동 2차원 힘 ∈ [−1, 1](env가 clip), 내부 frame_skip 5, 결정론적(Euler) |
| 데이터 | OSF `point_maze`의 `states.pth`(sha256 `e0c4e458…6990077`), `actions.pth`(`38a0b614…358e56`), `seq_lengths.pth`(`66333840…6345e20`): 궤적 2,000개 × 100 step. 프레임은 env가 그린다(obses는 쓰지 않음). DINO-WM 로더 분할(`get_train_val_sliced`, split_ratio 0.9, seed 42) → train 1,800 / valid 200 |
| 재생 검증 | 데이터 행동을 환경에 다시 넣으면 기록 상태를 **오차 0**으로 재생한다(2026-10-31, 궤적 50개 × 25 step, step별 오차 0). 환경이 곧 데이터 생성기다 |
| 월드모델 학습 분할 | 체크포인트가 로더 기본 seed(42)로 학습됐다고 가정하고 valid 200개를 "월드모델 미학습" 주 평가 집합으로 쓴다(가정을 기록). 보조 trainsplit 집합은 학습 데이터다 |

### 12.3 실패 거리 e, τ, 구간
- **e = ‖(x, y)_최종 − (x, y)_목표‖₂**(상태 단위). **τ = 0.5**: 환경 평가기 `eval_state`의 성공 문턱(DINO-WM의 PointMaze 성공 정의)과 같다. **r = e / 0.5**. 성공 r < 1 ⇔ `eval_state`의 success다(r과 성공 판정이 일치).
- 구간은 2절 그대로다(성공 r < 1, near 1 ≤ r < 2, mid 2 ≤ r < 3, far r ≥ 3). 정의는 하나다(PushT의 r_obj에 해당하는 보조 정의가 없다). 속도 차 ‖Δ(vx, vy)‖는 서술용으로 기록만 한다.
- τ의 화면 크기: 0.5 단위 ≈ 인코더 입력 196 px에서 약 1.4 패치(추정; 스모크에서 실측해 보고에 적는다). PushT의 20 px(≈ 0.55 패치)보다 크다.

### 12.4 episode·목표·상상 (E1 M1 프로토콜 그대로)
- episode = (궤적, offset) 균등 표집, 행동열 T = 5H step(데이터 행동을 로더 평균·표준편차로 역정규화한 원 단위). 시작 관측 = 시뮬레이터가 그린 1프레임(224 px) + proprio(상태 4차원). **목표 = 전문가(데이터) 행동열의 시뮬레이터 결과 상태·프레임**(r = 0; DINO-WM `goal_source=dset` 규약, E1 M1과 같음). 공식 PointMaze 계획 평가는 `random_state` 목표를 쓰지만 B는 계획을 하지 않으므로 목표는 평가 쌍 정의에만 쓰인다(기록).
- 상상: open-loop DINO-WM rollout(1프레임 시작 관측, 행동 정규화 후 5 step씩 묶음, H 단계) → 최종 latent. 실제: 시뮬레이터 최종 프레임(224 px, OSMesa 렌더) → DINOv2 인코딩. 수치 설정은 E1과 같다(SDPA attention, TF32).
- **horizon H ∈ {3, 5, 7, 10}, 주 칸 H = 5**(DINO-WM goal_H = 5). 궤적 길이 100이라 모두 가능하다.

### 12.5 반사실 severity grid (E1 M1과 같은 이분 탐색)
- 반사실 = 전문가 행동열 + s × 패턴. 패턴은 offset(무작위 단위벡터, 구간 내내 일정)과 iid(step마다 N(0, I))를 반씩. s의 단위는 힘(행동) 단위이고, env가 [−1, 1]로 clip한다.
- 표적 r*: 칸마다 3,600개 = [0.8, 3] 층화 균등 3,000 + [0.2, 0.8) 300 + [3, 6] 300. 표적마다 s를 이분 탐색 14회로 정한다. 도달하지 못한 표적은 새 episode로 넘긴다(최대 30회). episode마다 표적 6개와 전문가 행동열(r = 0) 하나.
- 상한 S_MAX(H, 패턴): 보정 집합 C(300 episode)에서 격자 S_GRID = {0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0}로 r의 중앙값을 재고, 중앙값이 6.5에 닿는 가장 작은 격자값(없으면 3.0). 이분 탐색 점검(400 episode: 도달률, |r − r*|)을 함께 한다. 평가 집합 상상 전에 고정.
- 목표: 칸마다 near-miss 1,000개 이상, 궤적 3,000개 이상(6.0절). 벽 때문에 도달 불가가 많으면 도달률을 보고한다.

### 12.6 분할과 칸
- 로더 train 1,800개를 seed 20261101 순열로 **J 800 / C 400 / T 600**(서로 겹치지 않음)으로 나눈다. J는 판별기 학습, C는 보정(S_MAX와 교란 크기), T는 보조 평가(trainsplit). **주 평가는 valid 200개**다. 판별기 학습·보정·평가의 시작 상태는 궤적이 달라 겹치지 않는다.
- 칸 = (집합, H): val × {3, 5, 7, 10}(주) + trainsplit × {3, 5, 7, 10}(보조) = 8칸. **판정 칸: val, H = 5, r, 판정기 (a) 앙상블.**

### 12.7 판정기 (평가 집합을 상상하기 전에 고정)
- **(a)**: E1과 같은 PairJudge([z, z_g, z − z_g] 토큰 → 2층 d = 128 → CLS; C = 384, P = 196), 라벨 r < 1, **K = 5 seed(0~4) 앙상블**(확률 평균 ≥ 0.5). 학습 집합(J): H마다 500 episode × (전문가 + 교란 7개) = 16,000 행동열. 교란 혼합은 E1과 같다(iid 0.35 / offset 0.35 / scale 0.15 / gross 0.15, 크기 = C에서 정한 50% 성공 크기 × log-U[0.3, 3]). 크기 보정 격자(힘 단위): iid σ {0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5}, offset β_max {0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0}, scale k_max {0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0}(E1과 같음), gross σ_big {0.3, 0.5, 0.7, 1.0, 1.5, 2.0}(σ_small 0.02; P(r ≥ 3) ≥ 0.9인 최소값). episode의 20%를 held-out으로 뺀다(RandomState(0)).
  - **사용 조건: 앙상블의 실제 프레임 held-out 정확도 ≥ 0.95.** 미달이면 E7을 멈추고 보고한다(기준을 낮추지 않는다).
- **상한(probe) 판별기 ai**: 같은 구조를 상상 최종 특징(H 넷 합침)으로 학습, K = 5. H4b용(규칙 대상 아님).
- **(b)**: d_vis 문턱 τ_b = 판별기 학습 부분(실제 프레임)에서 균형 정확도를 최대로 하는 값. PointMaze에는 사전 고정값이 없어 여기서 정하고 평가 전에 고정한다. 보조로만 보고하고 판정에 쓰지 않는다.
- 지각 바닥(실제 프레임의 near 오판)을 먼저 보고한다.

### 12.8 수량, 판정, 통계
- 분석은 `e1_m1_analyze.py`를 과제 전환으로 그대로 쓴다: 구간 확률, log r 로지스틱(전체 → r50·r10, 실패만 → 기울기), 시작 상태 무작위 절편 GLMM, AUROC(실제 / 상상 / probe), **출처 궤적 cluster bootstrap 1,000회**(seed 20261101), horizon 기울기(H마다 같은 재표집).
- **H1b(E7)** — 판정 칸(val, H5, r, (a))에서 세 조항 모두 성립할 때 성립: ① ΔAUROC(성공 대 near) = AUROC_실제 − AUROC_상상 ≥ 0.05, ② ΔAUROC(성공 대 far) ≤ 0.02, ③ [AUROC_상상 − AUROC_실제](성공 대 near)를 H ∈ {3, 5, 7, 10}에 회귀한 기울기의 bootstrap 95% 구간 상한 < 0.
- **H4b(E7)** — 같은 칸에서 (a)의 실제 AUROC(성공 대 near) ≥ 0.95이고, 상상 특징 probe(ai)의 AUROC와의 차 ≥ 0.05(점추정). v3.1 §9 E4b 정의와 같다.
- **서술(판정 없음)**: 지각 바닥, P(상상 성공 | near) [CI], P(상상 성공 | far), near/far, 실패(r ≥ 1)만의 log r 기울기, **양방향 이동**(P(상상 성공 | 성공) − 실제, P(상상 성공 | near) − 실제), r50·Δr10, (b), trainsplit 반복, 앙상블 구성원 SD, 표적 도달률. H1a형 구간 기준(near ≥ 0.3 등)은 참고로만 적는다(E7은 H1a 판정 대상이 아니다).
- **규약 점검(판별기 학습 전, 서술)**: val H5 전문가 episode 200개에서 d_vis(상상 최종, 실제 최종) / d_vis(실제 시작, 실제 최종)의 중앙값을 보고한다. 중앙값 ≥ 1(상상이 "가만히 있기"보다 나쁨)이면 월드모델과 시뮬레이터의 행동 규약이 어긋난 것이므로 멈추고 보고한다(Wall에서 본 종류의 불일치를 막는 점검).
- seed: 분할 20261101; episode seed는 E1 상수 + 700,000(보정 sweep 705000+H, 점검 705100+H·705200+H, 크기 705300+H, worker 707000/707500/707900 + 100H + i; 판별기 집합 episode 708000+H, 혼합 708100+H); 평가 칸 cell_seed = crc32('pm_<집합>_<H>_r') mod 100000(batch seed = cell_seed × 1000 + batch, worker seed = (cell_seed × 100003 + batch × 1000 + i) mod (2³² − 1)); 판별기 seed 0~4; bootstrap 20261101.

### 12.9 비용·시간 추정과 중단 규칙
| 단계 | 규모 | 추정 |
|---|---|---|
| pod 준비(apt, uv venv, mujoco-py 빌드, OSF 체크포인트 953 MB, DINOv2) | 1회 | 30~45분 |
| 보정(CPU, 렌더 없음; 재생 25 step < 1 ms) | H 4개 × (300 + 400 + 400) episode | < 5분 |
| 판별기 학습 집합(렌더 7.4 ms/프레임, 인코딩·상상) | 16,000 행동열 | 5~10분 |
| 판별기 10개(a 5 + ai 5), 30 epoch | 각 1.28만 행동열 | 25~40분 |
| 곡선 8칸 | 칸당 표적 3,600 + 전문가 약 600 | 20~30분 |
| 분석(CPU, bootstrap 1,000 × 8칸, GLMM) | | 5~10분 |
- 합계 약 **2.5~3 pod-시간 ≈ $1.5~2(A40)**, 4090이면 약 $3. 계획 상한 $20(cap), 감시 $28, 사용자 상한 $30.
- **중단 규칙**: ① (a) 앙상블 held-out < 0.95, ② 규약 점검 실패, ③ 누적 비용이 $20에 닿음(그때까지의 결과만 보고), ④ mujoco-py/OSMesa 설치가 1시간 안에 안 됨(대안 환경 없이 멈추고 보고). 어느 경우에도 기준을 낮추거나 과제를 바꾸지 않는다.

### 12.10 산출물과 기록
- 코드: `main/code/server/`의 E1 스크립트에 과제 전환(`B_TASK=pointmaze`; 경로·GPU 가드의 환경변수화)만 더한다. pod 준비·실행 스크립트는 `main/code/runpod/e7/`.
- 결과: `main/results/E7_runpod/pm/`(E1과 같은 파일 구성: `split_e7.json`, `calib_e7.json`, `judgeset_e7_meta.json`, `judges_e7.json`, 칸별 `eval_<집합>_H<H>_r.jsonl`(+ `_meta.json`), `e7_analysis.json`(E1 스키마: `cells`(`curve`, `ci`), `verdict_E7`), 그림). 작업 기록 `main/worklogs/E7_runpod.md`, 보고 `main/reports/E7_runpod.md`, LOG.md.
- 이 절의 sha256을 LOG.md에 적은 뒤에 pod를 만든다. 이 절에 없는 선택은 모두 편차로 기록한다.

---

## B. 시간순 기록

| 시각 | 작업 |
|---|---|
| 10:30 | 원고 세션의 지시 수신(E7: 두 번째 공개 DINO-WM 과제, 3단계, 단계마다 승인). DESIGN_main §0.2·§4·v1.1~§11, E1_server.md §1~3·§5~7, E1 코드 6개, bpaths·benv·r12lib·r21lib, A의 E0_dinowm.md와 wmlib.py를 읽음 |
| 10:35 | 서버 읽기 전용 점검: GPU 0 14.6 GB·100%, GPU 1 9.2 GB·99% — 둘 다 사용자의 `imlewm.train`(tmux e16/q0/q1). 체크포인트 `~/pmx/data/dinowm/outputs/{pusht,wall_single,point_maze}`, Wall 메타 `wall/wall_single/*.pth`, PointMaze 메타 `point_maze/*.pth` 확인. 디스크 여유 243 GB |
| 10:40 | DINO-WM 환경 코드 읽음(`env/wall/*`, `env/pointmaze/*`, `datasets/*`, `preprocessor.py`, `visual_world_model.py`). 인코더 입력은 224 → 196 px(패치 14 × 14 = 196). Wall `eval_state` 문턱 4.5 px, PointMaze 0.5 단위 |
| 10:45 | Wall CPU 점검(b-dinowm venv): 환경은 결정론적(벽 접촉 noise는 상수 0.5), 25 step 재생 8 ms, 렌더 1.6 ms. **데이터 규약 확인: s[t+1] = s[t] + a[t](1배, 잔차 1e-6), 환경 코드는 2배.** 1배로 고친 전체 재생은 1,920개 중 14%만 일치(접촉 처리가 다름). 궤적 길이 50 → H ≤ 9 |
| 10:50 | PointMaze CPU 점검(A의 a-dinowm-pm venv, 읽기 전용): 데이터 재생 오차 0(궤적 50 × 25 step), 렌더 7.4 ms/프레임, 100 step 궤적 2,000개, 25 step 순이동 중앙값 1.31 단위(문턱 0.5) |
| 10:55 | RunPod 도구 읽음(`runpod_rss/` budget·pod_up·sync; A RSS 실험이 `arx` 접두어로 진행 중, 잔액 $92.26). 사용자 지시: **E7은 서버가 아니라 RunPod에서, 예산 $30** |
| 2026-10-31T11:04Z | 위 A절 초안 작성, 사용자 승인 요청. 설계 문서·서버·RunPod는 건드리지 않음 |
| 11:13 | **사용자 승인("승인, 진행")**. §12를 DESIGN_main.md에 덧붙임(sha256 3273f801e657905daf08d20bab44ce3cf3c0db64e6a7340978c20f0cafd1de2f). LOG 기록. 2단계 시작: RunPod 도구 사본(`workspace/runpod_e7`, 접두어 bx7), E1 코드 과제 전환 패치, pod 스크립트 |
| 11:14~11:23 | 2단계 준비(맥북): `workspace/runpod_e7/`(runpod_rss 사본, 접두어 bx7, LIMIT $28, cap e7 $20, HARD_DEADLINE 11-07) 생성. E1 코드에 과제 전환 패치(`bpaths.py` 환경변수화 B_PLACE/B_TASK/B_PMX, `r21lib.py` pointmaze 과제, `r12lib.py` margin_parts_pm·proprio_of·sim_run_pm·ACT_DIV·SIG_SMALL·DEFS, `e1lib.py` 분할·seed·base-index 매핑, `e1_calib.py` S_GRID_PM·GRIDS_PM, `e1_judgeset/judges/m1_eval/m1_analyze.py` 접미사 e7·판정기 목록·(b) τ 고정·verdict_E7). 새 파일 `e7_smoke.py`, `e7_convcheck.py`, `e7_figs.py`, pod 스크립트 `main/code/runpod/e7/{e7_env.sh, setup_pod_e7.sh, run_e7_chain.sh}`. PointMaze 메타 .pth 3개를 서버에서 맥북 scratchpad로 복사(sha256 §12.2와 일치) |
| 11:25 | **pod 생성**: `bx7-e7`, NVIDIA A40(secure, $0.59/h), id r9wyi7bs4icy4h, 자동 종료 17:25Z(TERM_H 6). 생성 직전 잔액 $90.82(START_BALANCE). 감시 `budget.py watch 300` 시작 |
| 11:26 | **서버 CPU 드라이런**(A의 a-dinowm-pm venv, `~/pmx/B/tmp/e7_dry`, GPU 없음): 분할 1,800 → J 800/C 400/T 600(겹침 없음), val episode 표집(base index 매핑), 목표 재생이 기록 상태와 1.3e-6 일치, 이분 탐색·교란 계열 동작, 재생 0.8 ms·렌더 7 ms. **관찰**: U-maze에서는 r이 약 3.5에서 포화한다(offset s ≥ 0.7에서 중앙값 r 3.0~3.7, max r ≥ 6인 episode 3%). S_MAX는 규칙대로 격자 끝 3.0이 될 것이고, far 꼬리 [3, 6]의 표적 중 r* ≳ 3.7은 30회 재배정 뒤 버려질 것이다(도달률·버림 수를 보고; §12.5 규칙 범위 안) |
| 11:30 | pod에 코드(`/workspace/pmx/B/code`)와 데이터(`/workspace/pmx/data/dinowm/point_maze`) 전송, `setup_pod_e7.sh` 시작(apt, uv venv torch 2.4.1+cu124, mujoco-py 2.1.2.14 OSMesa, d4rl, dino_wm 0a9492f, OSF 체크포인트, DINOv2) |
| 11:34 | **pod 설치 끝**(약 5분): APT, torch 2.4.1+cu124, mujoco-py 2.1.2.14(OSMesa) 빌드·import OK, d4rl OK, dino_wm 0a9492f, OSF 체크포인트 sha256 `78c654ba…`(§12.2와 일치), hydra.yaml `25acae83…`, 데이터 .pth 3개 해시 일치, DINOv2 OK. nproc 96, A40 46 GB |
| 11:35 | **체인 시작**(`run_e7_chain.sh`, tmux 없이 setsid nohup; 로그 `/workspace/logs/`). 스모크 통과(재생 오차 ≤ 1e-4, 렌더 224×224, 특징 (1, 196, 384)) |
| 11:38 | 보정 끝(약 2분, CPU). 규약 점검 끝(11:39, 통과: 아래 수치) |
| 11:46 | 판별기 학습 집합 끝(7분, 16,000 행동열). 판별기 10개 학습 12:08 끝(21분) |
| 12:08 | **판별기 (a) 앙상블 held-out 정확도 0.9658 ≥ 0.95 → 사용 조건 통과**(규칙 12.7). 평가 집합 상상 시작: val H5 → H3 → H7 → H10 → trainsplit 4칸(칸당 약 3.5분) |
| 12:08~12:38 | **곡선 8칸 완료**(val H5·3·7·10 → trainsplit H5·3·7·10, 칸당 175~244초). 표적 3,600개 중 버림 1~38개(H3에서 많음), 시도당 도달률 0.43~0.65, near 1,335~1,366개·far 266~400개(기준 충족) |
| 12:38~12:45 | **분석 완료**(411초, bootstrap 1,000회) + 그림. `VERDICT E7`: ΔAUROC near 0.197 [0.172, 0.225] ✓, ΔAUROC far 0.076 [0.057, 0.097] ✗(≤ 0.02 위반), horizon 기울기 −0.033 [−0.037, −0.028] ✓ → **H1b(E7) 불성립**; 실제 AUROC 0.980 ≥ 0.95, probe 0.778(차 0.203) → **H4b(E7) 성립**. `E7_CHAIN_DONE` 12:44:49Z |
| 12:45 | 결과 회수(`pull_e7.sh` 방식: `results/E7_runpod/pm/` 52 MB — 분석 JSON, 원자료 JSONL 8개, 판별기 가중치 `judges_e7.pt` 23 MB, 그림 4장; pod 로그 `pod_logs/`). JSONL 줄 수 = 반사실 + 전문가 수와 일치 |
| 12:47 | **pod 삭제**(`pod_down.sh e7`, 장부 `gone`). 장부 비용 **$0.83**(1시간 22분 × $0.59 × 1.03), 계정 잔액 $86.82 |
| 12:50 | 점검 중 발견: `e1_calib.py`의 `sweep_job`에 PushT 단위 나눗셈 `/ 100.0`이 남아 있어 `calib_e7.json`의 sweep 표(중앙값 r)가 severity 척도 1/100에서 기록됨. S_MAX 결정값은 규칙상 격자 끝 3.0으로, 올바른 척도에서도 중앙값이 6.5에 닿지 않아 같다(드라이런 11:26, 재계산 아래). 이분 탐색·평가·판별기 집합은 패치된 `E.counterfactual`·`R.make_actions`를 써서 영향 없음. 코드를 고치고 서버 CPU에서 같은 seed로 sweep을 다시 계산해 `calib_e7_sweep_check.json`으로 둠 |
| 12:50~13:00 | 보고서 `reports/E7_runpod.md` 작성(판정 표, 모든 칸, 인용 수치·JSON 경로, 편차, 비용). 수치는 `summ_e7.py` 출력과 JSON에서 대조. `e1_calib.py` sweep 단위 수정, `e7_sweep_check.py`로 서버 CPU 재계산(11초): **S_MAX 모든 H·패턴 3.0 동일**(`pm/calib_e7_sweep_check.json`) |
| 13:02 | LOG.md에 E7 완료 기록. 공유 노트·메모리 갱신. 커밋은 하지 않음(지시 없음). 남은 작업·pod 없음 |
| 15:46 | 사용자 지시로 **커밋**: 예산 감시 프로세스 종료(pod는 12:47Z부터 gone, 추가 비용 없음). E7 변경 파일(설계 §12, LOG, 서버 코드 패치·신규 `e7_*.py`/`summ_e7.py`, RunPod 체인 `code/runpod/e7`·예산 도구 `workspace/runpod_e7`, 결과 `results/E7_runpod/`(판별기 가중치 `judges_e7.pt` 23 MB 포함), 보고, 이 작업일지)을 B 저장소 `main`에 커밋. `paper/README.md`·`paper/tmlr/`는 원고 세션 몫이라 제외. push 없음 |
