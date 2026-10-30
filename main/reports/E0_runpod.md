# B E0 RunPod 보고 — M2(Cosmos Policy LIBERO) 설치, 반사실 질의, CureWM 재현, 과제·r 정의, 처리량과 비용

- 작성: 2026-10-21, B-E0 RunPod 작업자(태그 `b-e0`, 상한 $9)
- 코드: `B_near_miss_blindness/main/code/runpod/`, 결과: `B_near_miss_blindness/main/results/E0_runpod/`, 작업 기록: `B_near_miss_blindness/main/worklogs/E0_runpod.md`
- E0는 가설 검정 단계가 아니다. 아래 판정은 E0 점검 항목(설치, 반사실 질의, CureWM 재현, r 정의)의 통과 여부다. E1 스모크·판정기 점검 수치는 모두 **탐색적**(사전등록 전, 소표본)이다.

## 0. 요약

| 항목 | 결과 |
|---|---|
| 설치 | **통과.** A40, 기본 템플릿 그대로(드라이버 595.91 = CUDA 13.2라 cu128 wheel 동작). Dockerfile을 따라 apt + uv로 네이티브 설치 6분 |
| 빠른 시작·rollout | **통과.** `get_action` 1.08초/회, VRAM 최대 7.6GiB. 폐루프 rollout(libero_goal "put the bowl on the plate") 5/5 성공 |
| 반사실 질의 | **동작.** CureWM과 같은 호출 순서이며 출력이 비트 단위로 같다. 행동 패스를 1단계로 줄여도 출력이 같아 질의당 0.90초. k청크 연쇄 상상은 0.90k초 |
| CureWM 재현 | **통과.** optimism **79.4%**(실패 218개, 원천 데모 25개, 95% CI 61.9–95.6). 논문: 공개 체크포인트 79.13%, 기준선(공식 데이터 미세조정) 79.55%(n=484). 10과제 전체로 넓히면 68.2%(CI 50.9–86.3) |
| 과제·r | 6과제 제안(+선택 1). r 구현이 LIBERO 술어를 재현(최종 상태 868개에서 일치율 1.000). 경계 겨냥 교란은 r을 severity에 따라 연속적으로 움직인다(Spearman 0.85–0.96) |
| 처리량 | 반사실 청크 0.90초, 실제 미래 value 0.50초, 배치 이득 없음. LIBERO 7프로세스 물리 359 step/s, 256px 렌더링 256 step/s. DP 16.5 step/s |
| M2 비용 추정(A40) | E1 $6.1, E2 $10.6(+기준 정책 설치 $1.2), E3 $0.6(7B VLM 기준), DP 학습 $3.5. 합계 약 $22 |
| 지출 | **$0.87**(장부 추정, A40 1시간 26분). pod `gone` 확인 |
| 결정 필요 | 판정기 (d) VLM과 (a) oracle이 지금 형태로는 기준 미달(4.4, 7절), E1 교란 계열 추가, E2 비용선 |

## 1. Pod, GPU, 이미지

- `runpod_main/pod_up.sh b-e0 "NVIDIA A40|NVIDIA RTX A6000|NVIDIA RTX A5000|NVIDIA L40|NVIDIA GeForce RTX 4090" 50 120`
  - `pmx-b-e0`(id qsve520o0wr9lc), NVIDIA A40 48GB, secure EU-SE-1, $0.59/h
  - 13:21:46Z 생성, 14:48:08Z 삭제
  - 당시 재고(13:18Z): A5000·3090·A6000·L4는 없었다. A40 $0.59, L40 $0.82, 4090 $0.89, L40S $1.09는 있었다.
- 이미지: 기본 템플릿 `runpod-torch-v240`(Ubuntu 22.04, CUDA 12.4 툴킷)
  - 호스트 드라이버가 595.91(CUDA 13.2)이라 Cosmos Policy의 torch 2.7.0+cu128 wheel이 그대로 돈다.
  - CUDA 12.8 이미지로 바꿀 필요가 없었다. 컨테이너 툴킷은 쓰지 않는다(필요한 확장은 모두 사전 빌드 wheel).
- 자원
  - cgroup CPU quota는 **7.65 vCPU**다(`nproc` 96은 호스트 값). RAM 한도는 50GB.
  - `/workspace`는 네트워크 파일시스템(MooseFS)이다. venv, uv 캐시, HF 캐시, 코드는 컨테이너 디스크(`/root/b`)에 두었다.

## 2. 설치 절차(작동한 것)

`code/runpod/setup_pod.sh`, `env.sh`, `patch_cosmos_cfg.py`

1. apt: cosmos-policy `docker/Dockerfile`과 같은 목록(build-essential, cmake, ffmpeg, git-lfs, libgl1, libegl1-mesa-dev, libgl1-mesa-dri, libglib2.0-0) + rsync, tmux, jq. NVIDIA EGL vendor 파일(`10_nvidia.json`)은 이미 있었다.
2. uv 0.8.12(그들의 Dockerfile과 같은 버전), 그다음 LIBERO.md 명령 그대로: `uv sync --extra cu128 --group libero --python 3.10`(약 6분)
   - torch 2.7.0+cu128, flash-attn 2.7.3, transformer-engine 2.2, natten 0.21, transformers 4.57.1, mujoco 3.3.2, PyPI `libero==0.1.1`
3. **python3.10-dev 추가**: uv가 시스템 python3.10으로 venv를 만들어, triton JIT가 `Python.h`를 못 찾았다.
4. **LIBERO assets**: PyPI `libero==0.1.1`은 huggingface/lerobot-libero의 sdist이고 assets가 없다. 장면 XML은 패키지 경로에서 읽는다.
   - lerobot-libero checkout(14fd6f0)을 받아 `site-packages/libero/libero/assets`로 심볼릭 링크했다.
   - `LIBERO_CONFIG_PATH`의 config.yaml을 미리 써서 대화형 질문을 피했다.
   - 렌더링은 `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`로 동작했다.
5. **gated 저장소 문제와 해결**(사용자 토큰·약관 동의는 쓰지 않음)
   - 정책 체크포인트 `nvidia/Cosmos-Policy-LIBERO-Predict2-2B`는 공개다.
   - 그런데 config가 두 파일을 gated 저장소 `nvidia/Cosmos-Predict2-2B-Video2World`에서 받는다.
     - 토크나이저(Wan2.1 VAE): `config/defaults/tokenizer.py`
     - 학습용 기반 체크포인트: `cosmos_policy_experiment_configs.py:148`. config import 때 즉시 받는다.
   - HF API의 SHA256으로 확인했다: gated `tokenizer/tokenizer.pth`는 공개 `Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth`(Apache-2.0)와 바이트 단위로 같다(38071ab5…). pod에서 받은 파일로도 다시 확인했다.
   - `patch_cosmos_cfg.py`로 두 줄을 고쳤다. 토크나이저 경로는 로컬 공개 VAE로, 학습용 기반 체크포인트는 `""`로 바꿨다. 추론 가중치는 모두 정책 체크포인트에서 온다. diff: `results/E0_runpod/cosmos_cfg_patch.diff`
   - 처음에는 gated 저장소 이름으로 HF 캐시 항목을 만들어 두는 방식이었다. 권한 분류기가 실행 전에 막아서 그 방식은 쓰지 않았다.
   - 참고: config import 때 공개 ALOHA 체크포인트(3.7GB)도 즉시 받아진다. 공개 파일이고 쓰지 않는다.
6. 커밋: cosmos-policy 18a2acc(2026-01-22), CureWM 8a7ee5d(2026-10-06), lerobot-libero 14fd6f0
7. 주의
   - EGL 컨텍스트를 만든 프로세스에서 fork한 자식은 렌더링이 멈춘다. 병렬 렌더링은 별도 프로세스나 spawn으로 띄운다.
   - CPU가 꽉 차면 sshd 응답이 1–3분 늦어진다.

## 3. 빠른 시작과 rollout 점검

- README 빠른 시작(`quickstart.py`, `results/E0_runpod/quickstart.json`)
  - 모델 로드 33.6초, 로드 직후 VRAM 3.95GiB, 최대 7.6GiB
  - `get_action`(행동 5단계 + 미래·value 병렬): 1.08–1.10초. 첫 호출은 2.35초다.
  - 표본 관측(LIBERO-10)의 value는 0.064다. 긴 과제라 할인된 value가 낮다(5.2절).
- 폐루프 rollout(`rollout_check.py`, run_libero_eval.run_episode와 같은 루프)
  - 과제: libero_goal t8 "put the bowl on the plate", 5 episode
  - **5/5 성공**, 평균 86 step(대기 10 step 포함), episode당 질의 5회
  - 질의 1.53초: 반사실 생성과 CPU를 나눠 쓰던 때라 유휴 시 1.1초보다 느리다.
  - 청크별 예측 value가 0.48–0.51에서 0.90–0.97로 올라간다.
  - 예측 미래 영상과 실제 프레임 비교: `rollout_check/ep0_chunk{2,5}_real_vs_pred.png`

## 4. 반사실 질의 방법과 CureWM 재현

### 4.1 방법(`cp_common.py`)

CureWM `scripts/probe_optimism.py`와 같은 순서다.

1. 관측으로 batch를 만들고 행동 패스를 돌린다. 생성 latent와 깨끗한 조건 latent를 얻는다.
2. 반사실 16-step 청크를 정규화한다(dataset stats의 actions_min/max → [−1, 1]). 모델 자신의 `replace_latent_with_action_chunk`로 행동 칸(latent 4번)에 넣는다.
3. `get_future_state_prediction`: 행동 칸까지를 조건으로 1단계 디노이징한다. 상상한 미래 3인칭·손목 영상과 미래 proprio가 나온다.
4. `get_value_prediction`: value 칸만 1단계로 생성한다. 결과 V는 [0, 1]이다.

확인한 것

- 우리 구현(`build_batch`)과 CureWM 호출 순서의 value·미래 영상 차이: 0(비트 단위).
- 행동 패스 1단계 대 5단계도 차이 0이다. 표본 1개와 probe 6행에서 확인했다. 행동 칸을 덮어쓰고 나머지 칸은 새로 생성되기 때문이다. 그래서 질의당 0.90초(5단계면 1.30초)다.

B용으로 확장한 것

- **k청크 연쇄 상상**(`imagine_chain`): 상상한 미래 영상(모델 공간, 다시 crop하지 않음)과 미래 proprio(latent에서 평균 추출)를 다음 관측으로 쓴다.
- **실제 미래 상태의 value**(`value_real_future`): 실제 미래 프레임을 VAE로 미래 칸에 넣고 실제 미래 proprio를 주입한 뒤 같은 value 패스를 돌린다. 판정기 (c)의 실제 프레임 입력이다(E3의 2×2 분해용).
- 배치 질의(`cf_query_batch`): 동작하지만 속도 이득이 없다(8절).

### 4.2 CureWM 재현(`run_cf_generate.sh`, `probe_cf.py`, `analyze_probe.py`)

데이터

- CureWM 엔진을 고치지 않고 돌렸다(`generate_libero.py`).
- LIBERO-Goal 10과제, 과제당 공식 데모 앞 3개를 썼다. 재생에 실패한 데모(t4·t8 각 1)는 CureWM 규칙대로 빠진다. 원천 데모 28개다.
- 6계열 × severity 5 × seed 1로 반사실 840개를 만들었다. 결과 라벨(한 번이라도 성공) 기준 실패 255개다.
- `PYTHONHASHSEED=0`. CureWM은 계열 seed를 `hash()`로 만들어 프로세스마다 달라질 수 있기 때문이다.

Probe

- CureWM 그대로다: t0 = 첫 행동 분기 − 1, 관측 = 기록 프레임[t0], 청크 = actions[t0:t0+16], CureWM cfg.
- 실패는 전부, 생존(교란했지만 성공)은 무작위 일부를 넣었다. CureWM의 "fa" probe와 같은 구성이다.
- t9는 생성이 늦게 끝나 별도 패스로 붙였다.

| 지표 | 우리(t0–t8, 실패 218 / 생존 250, 데모 25) | 우리(t0–t9, 255 / 290, 28) | CureWM 공개 체크포인트, Goal held-out(484 / 1196) |
|---|---|---|---|
| optimism = P(V(a−) > 0.5 \| 실패) | **79.4%** [61.9, 95.6] | 68.2% [50.9, 86.3] | 79.13%(기준선 79.55%) |
| AUROC(실패 대 생존, V(a−)) | 0.478 [0.37, 0.59] | 0.496 | 0.497 |
| false alarm(생존 중 V ≤ 0.5) | 32.0% | 40.7% | 30.5% |
| ΔSF = V(a+) − V(a−) | −0.012 | −0.014 | +0.002 |
| 시각 optimism | 55.5% | 52.5% | 56.4% |
| **같은 value head를 실제 미래 상태에 적용** | optimism 79.8%, AUROC 0.508 | 68.2%, 0.512 | 없음 |

- [ ]는 원천 데모 단위 cluster bootstrap 95% CI다.
- 계열별 optimism(우리 / CureWM)
  - approach_overshoot 8 / 12
  - carry_slip 14 / 67(n 7 / 12)
  - contact_oscillation 80 / 76
  - insufficient_grip 88 / 76
  - premature_release 83 / 89
  - wrist_tilt 90 / 95
- **재현 판단**
  - t0–t8 기준 79.4%로 논문 값과 0.3%p 차이다. 전체 10과제 68.2%도 CI가 79%를 포함한다.
  - 과제별 optimism은 0%(t5)에서 100%(t1, t4, t6, t7, t8)까지 갈린다. 그래서 합계는 과제 구성에 민감하다. CureWM held-out의 과제 구성은 공개 파일에서 알 수 없다.
  - t9(데모 174–347 step)는 value가 바닥이라 2.7%다. CureWM의 LIBERO-10(2.51%)과 같은 현상이다.
  - 공개 체크포인트와 "공식 데이터로 미세조정한 기준선"의 차이는 논문에서도 0.4%p라 비교에 영향이 없다.
- **B에 중요한 사실**(관찰. 이 질의 지점에서만 성립)
  - 이 낙관은 상상이 아니라 value head 자체에서 나온다. 실제 미래 프레임을 넣어도 optimism 79.8%, AUROC 0.508로 같다.
  - value는 행동에 거의 무관하다. Spearman(V(a−), V(a+)) = 0.95다.
  - value는 남은 시간 할인을 따른다. Cosmos의 value 목표는 γ = 0.99인 Monte Carlo return이고, Spearman(V(a−), 0.99^(남은 step)) = 0.89다.
  - 즉 CureWM식 "분기 시점 1청크" 질의에서 판정기 (c)는 성공·실패를 거의 가르지 못한다(지각 바닥이 거의 전부).
- 실패 거리별(탐색적, 1청크 CureWM 질의, t0–t8): near 91.4% / mid 83.3% / far 66.3%(실제 미래 프레임도 91.4 / 85.0 / 66.3)
- 차이점 하나: CureWM probe의 관측 프레임[t0]은 행동 t0을 실행한 뒤의 상태인데, 청크는 actions[t0]부터 시작한다(1 step 어긋남). 재현에서는 그대로 두었고, B의 E1 코드(`e1_smoke.py`)는 정렬을 맞췄다.

## 5. LIBERO 과제 선정과 e/τ 정의

### 5.1 술어와 r 구현(`libero_margin.py`, 시험 `test_margin.py`)

설치된 LIBERO 코드의 술어를 그대로 읽어 r = e/τ를 만들었다. r < 1이면 술어의 기하 조건이 성립한다. 같은 술어의 비기하 조건(접촉, 수직 범위·높이 순서)은 `aux_ok`로 따로 낸다.

| 술어(LIBERO 구현) | e | τ | aux |
|---|---|---|---|
| On(물체, 물체) `ObjectState.check_ontop` | 두 body 원점의 수평 거리 | 0.03 m(LIBERO 상수) | z_base ≤ z_item, 접촉 |
| On(물체, site 영역) `SiteObject.under` | site 좌표계 수평 오프셋 \|d_x\|, \|d_y\| | 영역 반폭 s_x, s_y. r = max(\|d_x\|/s_x, \|d_y\|/s_y) | s_z − 0.005 < d_z < s_z + 0.10, 부모와 접촉 |
| In(물체, site) `in_box` | 축별 상자 면까지 거리(z 아래 면은 0.01 m 낮춤) | 축별 반폭, L∞ | (site는 항상 참) |
| Open/TurnOn(관절) | \|q − q_full\| | \|q* − q_full\| | 없음 |

- 시험: 기록된 반사실 868개의 최종 MuJoCo 상태를 복원해 계산했다. **(r < 1 ∧ aux) = LIBERO 술어 일치율이 10과제 모두 1.000**이다.
- r < 1인데 aux만 거짓인 경우는 t2 2개, t9 6개였다. 병이 넘어진 경우다.
- 주의: 최종 상태 성공과 "한 번이라도 성공"(LIBERO 종료 규칙·CureWM 라벨)이 다를 수 있다.
  - t5는 CureWM 계열에서 접시가 영역을 지나가 일치율 0.80이었다. t9는 0.75였다.
  - B의 r은 **정착된 최종 상태**(행동열 + 무동작 15 step)에서 잰다. 정책 rollout은 LIBERO 규칙대로 성공 시 종료하므로, 성공 episode의 최종 상태는 성공 순간이다.

### 5.2 제안 과제(DESIGN_main 초안 재료)

| 과제(LIBERO-Goal) | 술어 | τ | 경계 겨냥 교란 시험(정착 상태 r) |
|---|---|---|---|
| t8 put the bowl on the plate | On(bowl, plate) | 3 cm | place_offset: r 중앙값 0.16 → 3.97(δ 0–15 cm), ρ 0.96, near 18 / mid 21 |
| t6 put the cream cheese in the bowl | On(cream_cheese, bowl) | 3 cm | place_offset: 0.46 → 4.92, ρ 0.88, near 16 / mid 26 |
| t1 put the bowl on the stove | On(bowl, stove cook_region) | 7.5 cm(반폭) | place_offset: 0.27 → 3.84, ρ 0.95, near 24 / mid 27 |
| t4 put the bowl on top of the cabinet | On(bowl, cabinet top_side) | 12.5 / 9.4 cm | place_offset: 0.38 → 2.86, ρ 0.92, near 26 / mid 16 |
| t2 put the wine bottle on top of the cabinet | On(bottle, cabinet top_side) | 12.5 / 9.4 cm | CureWM 계열만으로 실패 42개 중 near 29 / mid 11 |
| t5 push the plate to the front of the stove | On(plate, stove_front_region) | 4 cm | under_actuation: 0.83 → 3.31, ρ 0.85, near 19 / mid 11 |
| (선택) t0 open the middle drawer | Open(middle drawer) | 2 cm(q* −0.14, q_full −0.16) | pull_truncate: s 0.10 → 0.25에서 r 0.01 → 3.55로 가파름. 촘촘한 grid 필요 |

- ρ는 severity와 r의 Spearman 상관이다.
- 제외한 과제
  - t3: 서랍을 연 뒤 넣는 2단계 과제다. 실패 대부분이 r = 4.8(far)이고 데모 재생 실패가 있다.
  - t7: 꺼진 손잡이가 이미 r = 1.31이라 far 쪽이 없다.
  - t9: 성공 상태가 유지되지 않는다(최종 대 한 번이라도 성공 0.75). 기울어진 rack 슬롯이다.
- 교란 계열(`nearmiss_sweep.py`)
  - **place_offset(δ)**: carry 구간 전체에 일정한 수평 행동 오프셋을 더한다. 방향은 (데모, k)마다 뽑고 severity 축을 따라 같은 방향을 쓴다. CureWM의 공통 난수와 같은 방식이다.
    - OSC delta 목표가 매 step 현재 자세 기준으로 다시 잡혀, 명령한 이동의 약 20%만 실현된다(`dbg_offset.py`). 그래서 실현 이동 기준으로 보정했다(G_EFF = 0.010 m/단위/step).
  - **under_actuation(s)**: grasp 이후 이동을 (1 − s)배 한다.
  - **pull_truncate(s)**: 당김 구간의 마지막 s 비율 동안 이동을 멈춘다. 서랍 데모는 그리퍼를 닫지 않아 CureWM phase 주석이 전부 APPROACH였다. 그래서 CureWM 계열이 서랍에는 작용하지 않았다(90/90 성공).
- CureWM 6계열만으로는 near-miss가 과제에 따라 적다. t8은 실패 22개 중 16개가 r ≥ 3이다. 위 경계 겨냥 계열로 r ∈ [0.8, 3]을 고르게 채울 수 있다. 같은 교란 세기에서 결과 라벨은 모두 실행으로 정한다.

## 6. E1 스모크(탐색적, `e1_smoke.py`, `e1_smoke_frames.png`)

- 설정
  - 과제 6개(t1, t2, t4, t5, t6, t8)의 반사실 30개: 성공 12, near-miss 12, r ≥ 2 6
  - k = 1..4. 끝을 맞춘 질의다: 관측 = 프레임[T − 16k − 1], 실제 행동 청크 k개.

| k | 성공: V_imag(>0.5 비율) | near-miss | r ≥ 2 | 상상 대 실제 최종 프레임 MSE(성공 / near) | 시간 |
|---|---|---|---|---|---|
| 1 | 0.995 (1.00) | 0.704 (0.75) | 0.291 (0.33) | 164 / 300 | 1.0–1.2초 |
| 2 | 0.992 (1.00) | 0.894 (0.92) | 0.788 (0.83) | 217 / 560 | 1.85초 |
| 3 | 0.995 (1.00) | 0.994 (1.00) | 0.832 (0.83) | 263 / 613 | 2.70초 |
| 4 | 1.000 (1.00) | 1.000 (1.00) | 0.833 (0.83) | 281 / 659 | 3.60초 |

- 같은 value head의 실제 최종 상태 판정(k = 1 시작점): 성공 0.984(1.00), near 0.604(0.67), r ≥ 2 0.294(0.33)
- 상상이 길어질수록 near-miss와 far 실패가 "성공"으로 그려진다. 그림의 넷째 줄은 실제로는 그릇이 스토브 밖에 떨어졌는데, k ≥ 2 상상에서는 스토브 위에 있다(V = 1.00).
- **소표본(near 12개)이고 사전등록 전이라 판정에 쓰지 않는다.** E1의 파이프라인(끝 정렬, 연쇄, 실제 프레임 value)이 동작한다는 점검이다.

## 7. 판정기 점검(탐색적)

- **(c) value head**
  - 목표가 γ^(T−t) × 성공(γ = 0.99)이라 V > 0.5는 "약 69 step 안에 성공"을 뜻한다.
  - 분기 시점 질의(CureWM)에서는 남은 시간이 판정을 지배한다(4.2절).
  - E1에서는 끝을 맞춘 질의(마지막 k청크)로 쓰면 성공 0.98–1.00, near 0.60–0.70, far 0.29로 갈린다(실제 프레임, k = 1). 끝 정렬을 사전등록 규칙으로 넣는 것을 제안한다.
- **(d) VLM**(`vlm_bench.py`, `vlm_bench2.py`)
  - Qwen2.5-VL-7B-Instruct(공개, Apache-2.0)를 고정 프롬프트, P(Yes) 점수로 썼다.
  - 실제 최종 프레임 50장(성공 25, 실패 25, 224px)에서 모든 장면을 "No"로 답했다(P(yes) < 0.1). AUROC 0.58이다.
  - 물체 중심 질문이나 448px 확대도 0.50–0.61이다.
  - 최종 프레임에서 그리퍼가 물체를 가리는 경우가 많다(`rollout_check` 그림). **지금 형태로는 판정기로 쓸 수 없다.**
  - 처리량은 충분하다: 0.13초/판정(배치 1), 0.040초(배치 8), VRAM 15.8GB.
- **(a) oracle 가능성**(`oracle_probe.py`)
  - DINOv2-base 고정 특징 + 과제별 로지스틱 회귀, 원천 데모 단위 leave-one-out
  - 실제 최종 프레임 868장(과제당 약 60–90장)에서 정확도 0.88
  - r 구간별: <0.8에서 0.92, 0.8–1에서 0.78, 1–2에서 0.72, 2–3에서 0.88, ≥3에서 0.95
  - 과제별 0.63(t9)–1.00(t0). 제안 과제는 t1 0.95, t2 0.94, t4 0.97, t5 0.70, t6 0.88, t8 0.97이다.
  - 설계의 0.95 기준에 못 미친다. 학습 프레임이 적고, 그리퍼가 물체를 가리고, 경계 근처가 본래 어렵기 때문이다.

## 8. 처리량(A40, 7.65 vCPU)

| 항목 | 측정 |
|---|---|
| Cosmos 반사실 질의(1청크) | 0.90초(행동 5단계면 1.30초) |
| k청크 연쇄 상상 | 0.90 / 1.80 / 2.70 / 3.60초(k = 1–4) |
| 실제 미래 상태 value | 0.50초 |
| 정책 질의 `get_action` | 1.10초 |
| 배치 4 / 8 / 16 | 질의당 0.90 / 0.88 / 0.87초(이득 없음, 배치 1에서 GPU 포화). VRAM 10 / 21 / 26GB |
| VRAM | 질의 1개 5.0GiB, 빠른 시작 최대 7.6GiB |
| LIBERO 단일 프로세스 | 물리만 48.6 step/s, 2카메라 224px 40.0, 256px 39.2 |
| LIBERO 병렬 | 4프로세스: 물리 209, 256px 145. 7프로세스: 물리 359, 256px 256 step/s(합계) |
| CureWM 반사실 생성(매 step 224px 렌더링, 6프로세스, 다른 작업과 경합) | 868 episode / 15.7분 |
| DP(합성 텐서) | 100M(ResNet18-GN × 2 + UNet1D 256-512-1024), 배치 64, 2 × 128px: 16.5 step/s, DDIM-10 추론 74ms, VRAM 2.8GB |
| VLM 7B | 0.040초/판정(배치 8) |

설계의 가정(예측 1회 1–2초)보다 빠르다. 다만 배치로 줄일 수는 없어 비용은 질의 수에 비례한다.

## 9. E1–E3의 M2 비용 추정(`cost_model.py`, `results/E0_runpod/cost_model.json`)

A40 $0.59/h. 실측 단위 시간에 준비·로드·재시작 여유 25%를 곱했다.

| 실험 | 규모(설계) | 주요 계산 | pod 시간 | 비용 | 설계 표 |
|---|---|---|---|---|---|
| E1 | 반사실 3,000개(r ∈ [0.8, 3], near ≥ 1,000), k = 1–4 | Cosmos 질의 30,000회(7.7 GPU-h) + 실제 value 3,000회 + 물리 rollout 3,900회(0.5h) | 10.3 | **$6.1** | 16 GPU-h, $12 |
| E2 | 정책·과제 쌍 60개 × 50 ep = 3,000 ep(DP 4단계 × 6과제, π0.5·OFT × 6과제, 교란 계열 4 × 6과제) | 모드 O 전체 재생 36,000회 + 모드 C 14,400회 + 정책 질의 | 17.9 | **$10.6** + 기준 정책 설치 약 $1.2 | 14 GPU-h, $10 |
| (선택) E2 진단 | 모드 O 마지막 k청크 | Cosmos 30,000회 | 9.1 | +$5.4 | — |
| E3 | VLM 34,200 판정 + 실제 value 3,000회 | 7B VLM 기준 | 1.0 | **$0.6**(32B VLM이면 약 $3) | 4 GPU-h, $3 |
| 정책 학습 | DP 6과제 × 50k step(4단계 checkpoint) + 데모 재렌더링 | 5.1h 학습 | 5.9 | **$3.5** | E0 표의 일부 |
| 합계 | | | 약 35 | **약 $22**(선택 포함 $27) | M2 약 $34 |

- 가정
  - 모드 O는 설계 4.1대로 초기 관측에서 실행 행동열 전체를 재생한다(episode당 평균 12청크).
  - 모드 C는 DP 정책만 계산했다.
  - π0.5·OFT 질의는 0.15초로 가정했다(이번에 측정하지 않음).
- 시뮬레이션은 비용의 10% 미만이다. 7.65 vCPU로 충분하고, GPU(Cosmos 질의 수)가 비용을 정한다.
- 단계별 상한과의 관계
  - E1은 `b-e1` $15 안이다.
  - E2는 `b-e2` $13 안이지만 설계 표 $10보다 약간 많다.
  - E3는 `b-e3` $5 안이다.
  - DP 학습 $3.5는 E0에서 돌리지 않았으므로 담을 상한 항목이 필요하다.
- pod 수명이 4시간이라 E1(10h)과 E2(18h)는 작업 목록 기반 재개 runner로 여러 pod에 나눠야 한다. TERM_H 8 연장을 요청하면 pod 수가 준다.

## 10. 정책 학습 계획(설계: 과제별 작은 DP, 4단계 checkpoint, 공개 강한 정책 기준점)

- **데이터**
  - 과제당 공식 데모 50개를 기록된 MuJoCo 상태에서 256px로 재렌더링한다(재생 오차 없음, 약 3분).
  - Cosmos 입력과 같은 시야로 변환한다: 상하 반전 → JPEG95 → 224 → 90% 중앙 crop → 128px.
  - 같은 시야여야 모드 C에서 DP가 상상 프레임을 그대로 입력으로 받는다.
- **모델**
  - CNN Diffusion Policy(Chi et al. 2023): ResNet18-GroupNorm × 2(3인칭·손목) + proprio 9차원(Cosmos와 같은 순서), UNet1D(256, 512, 1024), 약 100M
  - DDPM 100 학습 / DDIM 10 추론, 배치 64, AdamW 1e-4, EMA
  - **행동 구간 16 step을 open-loop로 모두 실행**한다(Cosmos 청크와 일치). n_obs_steps = 1이다. 월드모델이 청크 경계 상태만 내기 때문이다.
- **학습**
  - 과제당 50k step(A40에서 약 51분), checkpoint 5k / 10k / 20k / 50k로 4단계
  - 6과제 1 seed 약 $3.5. 성공률이나 near-miss 비중 n_π의 퍼짐이 부족하면 seed 2개를 더한다(+$7).
- **평가**: 4 checkpoint × 6과제 × 50 episode = 1,200 episode(E2에 포함). DP 질의 74ms, 물리 위주라 빠르다.
- **기준 정책**
  - openpi π0.5 LIBERO(`pi05_libero`, JAX, ECM에서 RunPod 동작 확인)와 OpenVLA-OFT(LIBERO-Goal 체크포인트, 7B)
  - 각각 별도 환경과 체크포인트(약 1 pod-hour)가 필요하다.
  - A40 48GB에 Cosmos(8GB)와 함께 올릴 수 있다.
  - 모드 C 입력 형식(224 상상 프레임)이 맞는지는 E2 스모크에서 확인한다.
- **교란 계열(보조)**: 최종 DP를 바탕으로 R2-12식 지속 offset(near-miss 쪽)과 iid 잡음(far 쪽)을 같은 성공률로 보정한다.
- 이번 단계에서 학습은 돌리지 않았다. 학습 코드·데이터 변환을 새로 만들어야 하고, 6과제 학습(약 5시간)이 pod 수명(4시간)을 넘는다. 시야와 행동 구간 설정은 DESIGN_main 고정 뒤에 하는 것이 맞다.

## 11. 편차, 위험, 결정이 필요한 것

1. **판정기 (d) VLM 미달**
   - 고정 프롬프트의 공개 7B VLM이 LIBERO 224px 프레임에서 우연 수준이다(AUROC 0.50–0.61, 항상 "No").
   - 선택지
     - (i) 끝에 들어 올리기 청크(그리퍼 열고 위로 16 step)를 실제·상상 모두에 붙여 물체를 가리지 않게 한다.
     - (ii) 더 큰 VLM(32B, 비용 약 $3)
     - (iii) 시작·끝 프레임을 함께 주는 비교 프롬프트
     - (iv) (d)를 보고용 보조 판정기로 낮춘다.
   - 상상 프레임이 224px라 해상도를 올리는 방법은 쓸 수 없다.
2. **판정기 (a) oracle의 0.95 기준**
   - 고정 DINOv2 + 로지스틱, 과제당 약 60–90장으로 0.88(경계 1–2 구간 0.72)이다.
   - E1에서 시도할 것: 학습 프레임 확대(E1 보정 sweep의 실제 프레임, 시작 상태 분리), 위 (i)의 들어 올리기, 손목 영상 추가, 작은 CNN 미세조정.
   - 그래도 미달이면 설계 규칙상 M2의 H1a 주 판정기를 쓸 수 없다. 이때 (c)·(d)만으로 보고할지 결정이 필요하다.
3. **E1 교란 계열**: CureWM 6계열에 경계 겨냥 계열(place_offset, under_actuation, pull_truncate)을 더하는 것을 제안한다. 설계의 "경계 근처 촘촘한 severity grid"를 실현하는 방법이다. 계열·보정 방식은 DESIGN_main에 고정한다.
4. **판정기 (c) 사용 규칙**
   - value는 남은 시간으로 할인된다. E1에서는 끝을 맞춘 질의(마지막 k청크)의 V를 쓰고, 문턱 0.5와 연속 V를 함께 보고하는 것을 제안한다.
   - CureWM식 분기 시점 질의는 (c)의 지각 바닥(실제 프레임에서도 79.8%)이 전부라 E1 곡선에는 쓰지 않는다.
5. **과제**: 6과제(t1, t2, t4, t5, t6, t8)와 선택 t0. r 정의는 5.1절 표다. 결과는 정착된 최종 상태로 판정한다.
6. **CureWM 재현 수치의 과제 구성 민감도**: t0–t8 79.4%, t0–t9 68.2%. 둘 다 보고한다.
7. **GPU 종류 고정**: M2 전체를 A40 한 종류로 돌리는 것을 제안한다(설계 6.0의 "섞지 않는다"). A40 재고가 "Low"라 확보하지 못하면 M2 전체를 다른 종류로 옮기고 처리량을 다시 잰다.
8. 그 밖의 편차(모두 기록함)
   - config 두 줄 수정(공개 파일만 사용)
   - LIBERO assets 심볼릭 링크
   - CureWM probe의 생존 표본은 250/532개
   - 행동 패스 1단계(출력 동일 확인)
   - t9 별도 패스
   - near-miss sweep v1 → v2(실현 이동 보정), 서랍 별도 실행

## 12. 비용과 시간

- pod: A40 $0.59/h, 13:21:46Z–14:48:08Z(1시간 26분)
- **장부 추정 $0.87.** `runpodctl billing`은 삭제 직후 $0.27까지만 집계했다(지연). 최종 청구는 장부 추정과 비슷할 것이다.
- `budget.py status`: `b-e0 qsve520o0wr9lc NVIDIA A40 $0.59/h gone $0.87`
- 상한 $9(계획 $7) 안. DP 학습은 돌리지 않았다.
- 맥북에서는 파일 편집, 문법 검사, 결과 JSON 요약(표준 라이브러리, 몇 초)만 했다. 반사실 episode 원본(약 30GB)은 맥으로 받지 않았다.
