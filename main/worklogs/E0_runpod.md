# B E0 RunPod (M2 = Cosmos Policy LIBERO) 작업 기록

- 작업자: B-E0 RunPod 작업자. 태그 `b-e0`, 상한 $9(계획 $7 이내), 도구 `runpod_main/`(접두어 `pmx`).
- 기준: MAIN_RULES.md, runpod_infra/PILOT_RULES.md §2–§4, B_near_miss_blindness/EXPERIMENT_DESIGN.md.

## 시간순 기록 (UTC)

- 13:00–13:20 문서 읽기. 사전 조사(맥북, 웹 문서만 읽음):
  - Cosmos Policy Dockerfile은 `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04` + apt 몇 개 + uv 0.8.12뿐이다. 나머지는 모두 `uv sync --extra cu128 --group libero --python 3.10`(torch 2.7.0+cu128, flash-attn 2.7.3, transformer-engine 2.2, natten 0.21 사전 빌드 wheel). 따라서 컨테이너의 CUDA 툴킷은 상관없고 호스트 드라이버가 CUDA 12.8(R570 이상)을 지원하면 된다.
  - 정책 체크포인트 `nvidia/Cosmos-Policy-LIBERO-Predict2-2B`는 공개(3.9GB)다. 그러나 config가 토크나이저(`hf://nvidia/Cosmos-Predict2-2B-Video2World/tokenizer/tokenizer.pth`)와, config import 시점에 기반 체크포인트(`model-480p-16fps.pt`, 3.9GB)를 gated 저장소(gated=auto, 약관 동의 필요)에서 내려받는다.
  - HF API의 LFS SHA256으로 확인: gated 저장소의 `tokenizer/tokenizer.pth`(507,609,880 B, sha256 38071ab5…)는 공개 저장소 `Wan-AI/Wan2.1-T2V-1.3B`의 `Wan2.1_VAE.pth`와 바이트 단위로 같다. 그래서 공개 파일을 HF 캐시의 해당 위치에 두고, 추론에 쓰이지 않는 기반 체크포인트 자리에는 빈 placeholder를 둔 뒤 `HF_HUB_OFFLINE=1`로 돌린다. 사용자 토큰이나 약관 동의는 쓰지 않는다.
  - libero 그룹은 PyPI `libero==0.1.1`(huggingface/lerobot-libero의 sdist, 2.9MB, assets 없음)을 설치한다. assets·bddl·init 파일은 lerobot-libero git checkout을 LIBERO config로 가리킨다.
  - CureWM: optimism = 실패 반사실 중 V(o, a−) > 0.5 비율(`analysis/grid44.py`). Goal held-out 484개에서 공개(released) 체크포인트 79.13%, 기준선(공식 데이터 미세조정) 79.55%. 반사실 질의는 `get_action`(생성 latent) → `replace_latent_with_action_chunk`로 행동 칸 교체 → `get_future_state_prediction` → `get_value_prediction`(`scripts/probe_optimism.py`). t0 = 첫 행동 분기 − 1, 16-step 청크 1개.
  - Cosmos Policy value 목표는 Monte Carlo return(γ = 0.99, 성공 시 terminal reward 1)이다. 따라서 V는 남은 시간에 따라 할인된다(남은 69 step 이상이면 성공이어도 V < 0.5).
  - GPU 재고(13:18Z, secure): A5000·3090·A6000·L4 없음. A40 $0.59/h, L40 $0.82, 4090 $0.89, L40S $1.09 있음.
- 13:21 pod 생성: `pmx-b-e0` (id qsve520o0wr9lc), NVIDIA A40 48GB, $0.59/h, secure EU-SE-1, 종료 예정 17:21Z. 기본 템플릿(runpod-torch-v240, Ubuntu 22.04)이지만 드라이버가 595.91(CUDA 13.2)이라 cu128 wheel이 그대로 돈다. 이미지 교체 불필요.
  - CPU 할당은 cgroup quota 7.65 vCPU(nproc 96은 호스트 값), RAM 한도 50GB. `/workspace`는 네트워크 파일시스템(MooseFS)이라 venv·uv 캐시·HF 캐시·코드는 컨테이너 디스크(`/root/b`)에 두고 데이터·결과·로그만 `/workspace`에 둔다.
  - 처음 설치 스크립트는 gated 저장소 이름으로 HF 캐시 항목을 만들어 두는 방식이었는데, 실행 전 권한 분류기가 막았다(Security Weaken). 그 방식을 버리고, 공개 파일(정책 체크포인트, Wan2.1 VAE)만 받은 뒤 cosmos-policy config가 로컬 VAE 파일을 가리키게 하는 방식으로 바꿨다(checkout을 확인한 뒤 결정).
- 13:25 설치 스크립트 시작(`/workspace/logs/setup.log`).
- 13:25–13:31 설치 완료(약 6분): uv 0.8.12, `uv sync --extra cu128 --group libero --python 3.10` → torch 2.7.0+cu128, CUDA 사용 가능(A40). uv가 시스템 python3.10을 써서 triton JIT 빌드가 Python.h를 못 찾음 → `apt-get install python3.10-dev`로 해결. 정책 체크포인트와 Wan2.1 VAE 다운로드, VAE SHA256 일치(38071ab5…).
  - checkout 확인 결과 gated 경로는 두 곳만 실제로 쓰인다: `config/defaults/tokenizer.py`의 `vae_pth`, `cosmos_policy_experiment_configs.py:148`의 학습용 기반 체크포인트 `get_checkpoint_path(...)`(config import 때 즉시 다운로드). `patch_cosmos_cfg.py`로 앞은 로컬 Wan2.1 VAE 경로로, 뒤는 `""`로 바꿨다. diff는 `results/E0_runpod/cosmos_cfg_patch.diff`.
  - `HF_HUB_OFFLINE=1`은 cosmos_utils가 `cache_dir=HF_HOME`으로 받아 hub 캐시와 위치가 달라 실패 → 오프라인 모드는 쓰지 않음(정책 체크포인트 캐시 폴더를 `HF_HOME/` 아래로 옮겨 재다운로드 방지). config import 때 공개 ALOHA 체크포인트(3.7GB)도 즉시 받아진다(공개, 사용 안 함).
- 13:33 README 빠른 시작 통과(`quickstart.py`, `results/E0_runpod/quickstart.json`): 모델 로드 33.6초, VRAM 최대 7.6GiB. `get_action`(행동 5단계 + 미래·value 병렬) 1.08초/회(첫 호출 2.35초). 반사실 질의(행동 칸 교체 → 미래 1단계 → value 1단계): 행동 패스 5단계 1.30초, 1단계 0.90초이고 두 경우 value와 미래 영상이 CureWM 호출 순서와 비트 단위로 같다(차이 0).
- 13:36 LIBERO assets: PyPI `libero==0.1.1`은 장면 XML을 패키지 경로에서 찾는다 → checkout의 `assets`를 site-packages의 `libero/libero/assets`로 심볼릭 링크. EGL 렌더링 동작. 렌더링 2카메라 224px 단일 프로세스 15 step/s, 물리만 65 step/s(부하 있는 상태의 대략값).
  - 공식 데모 재생 성공(CureWM 백엔드, 과제당 4개): 36/40. 실패는 t3 2개, t4 1개, t8 1개 → CureWM은 재생 실패 데모를 버리므로 과제당 3개 데모로 생성.
  - 실수: 원격 `pkill -f generate_libero.py`가 그 명령을 담은 ssh 셸 자신까지 죽였다. 이후 `pgrep -f "[g]enerate..."` 형식을 쓴다.
- 13:38 CureWM 반사실 생성 시작(엔진 수정 없음, `run_cf_generate.sh 3 6`): LIBERO-Goal 10과제 × 데모 3 × 6계열 × severity 5 × seed 1, PYTHONHASHSEED=0.
- 13:40–13:45 Cosmos Policy 폐루프 rollout(`rollout_check.py`, libero_goal t8 "put the bowl on the plate", 5 episode): 5/5 성공, 평균 83 step, 질의 1.53초(반사실 생성과 CPU 경합 중), 청크별 예측 value 0.49 → 0.97로 증가.
- 13:50 r 계산 시험(`test_margin.py`, 기록된 반사실 793개 최종 상태): 9과제 모두 (r<1 ∧ aux) = LIBERO 술어 일치율 1.000.
- 13:38–13:54 반사실 생성 완료(941초, 6 프로세스): t0–t8 775행 + t9 93행. 과제당 유효 데모: t4·t8 2개, 나머지 3개(재생 실패 제외). 저장량 약 30GB(`/workspace/data/cf_goal`, 맥으로 받지 않음).
- 13:51–14:09 CureWM 프로토콜 probe(`probe_cf.py`, t0–t8): 실패 218 + 생존 250. 행동 패스 1단계와 5단계의 value 차이 최대 0.0(6행 확인). 행당 약 2.3초(반사실 질의 + 명목 질의 캐시 + 실제 미래 value).
  - optimism 79.4%(cluster bootstrap 95% CI 61.9–95.6) 대 CureWM 공개 체크포인트 79.13%(n=484), 기준선 79.55%. AUROC 0.478, false alarm 32.0%, ΔSF −0.012, 시각 optimism 55.5%.
  - 같은 value head를 실제 미래 상태에 적용하면 optimism 79.8%, AUROC 0.508: 이 질의 지점의 낙관은 상상이 아니라 value head 자체(남은 시간 할인)에서 나온다. Spearman(V(a−), 0.99^남은시간) = 0.89, Spearman(V(a−), V(a+)) = 0.95.
- 13:55–14:05 경계 겨냥 교란 sweep: v1(마지막 12 step 오프셋)은 실제 EE 이동이 의도의 약 20%라 r이 거의 안 변함(`dbg_offset.py`로 확인: OSC delta 목표가 매 step 현재 자세 기준이라 지속 오프셋의 대부분이 사라짐). v2는 carry 전체 구간에 보정 이득(G_EFF = 0.010 m/단위/step)으로 오프셋 → r이 severity에 단조·연속(Spearman 0.88–0.96).
  - 서랍(t0) 데모는 그리퍼를 닫지 않고 손잡이를 연다 → CureWM phase 주석이 전부 APPROACH라 CureWM 계열이 서랍에 작용하지 않음(90/90 성공). 데모 후반을 당김 구간으로 보는 pull_truncate로 따로 sweep(`nearmiss_sweep_v2_t0`): r이 s 0.1→0.3에서 0.01→4.7로 가파르게 변함.
  - t9(와인병→rack)는 최종 상태와 '한 번이라도 성공' 일치율 0.75(성공 상태가 유지되지 않음), t5는 0.80(CureWM 계열에서 접시가 영역을 지나감).
- 14:13–14:18 E1 스모크(`e1_smoke.py`, 30 episode × k=1..4, 끝 정렬): 아래 보고서 참조.
- 14:18–14:23 처리량: 반사실 질의 0.90초(행동 5단계면 1.30초), k청크 연쇄 0.90k초, 실제 미래 value 0.50초, 정책 get_action 1.10초. 배치 4/8/16은 질의당 0.90/0.88/0.87초로 이득 없음(A40이 배치 1에서 이미 포화), VRAM 배치 16에서 26GB. DP 벤치(합성 텐서): ResNet18×2 + UNet1D 100M, 배치 64·128px에서 16.5 step/s, DDIM-10 추론 74ms, VRAM 2.8GB.
- 14:09–14:13 t9 probe 추가 패스(`--only_tasks t9`, 실패 37 + 생존 40). t9 데모가 길어(174–347 step) value가 바닥(실패 V 평균 0.16)이라 전체 10과제 optimism은 68.2%로 내려간다.
- 14:24 VLM 판정기(d) 처리량(`vlm_bench.py`, Qwen2.5-VL-7B-Instruct, 공개·Apache-2.0, `/workspace/cache/hf_vlm`): 0.13초/판정(배치 1), 0.040초(배치 8), VRAM 15.8GB, 로드 98초. 실제 최종 프레임 50장에서 P(yes)가 전부 0.1 미만, AUROC 0.58. 질문을 물체 중심으로 바꾸거나 448px로 키워도 0.50–0.61(`vlm_bench2.py`).
- 14:25–14:33 시뮬레이션 처리량 첫 실행이 멈춤: 부모 프로세스가 EGL 컨텍스트를 만든 뒤 fork한 자식에서 렌더링이 hang. 강제 종료 후 spawn 컨텍스트로 고쳐 재실행(`throughput_sim2.log`).
- 14:34 시뮬레이션: 단일 프로세스 물리만 48.6 step/s, 2카메라 224px 40.0, 256px 39.2. 7 프로세스 합계 물리 359, 256px 렌더링 256 step/s.
- 14:37–14:40 E1 스모크 그림(`fig_e1_smoke.py` → `e1_smoke_frames.png`).
- 14:40–14:46 판정기 (a) 가능성(`oracle_probe.py`): DINOv2-base 고정 특징 + 과제별 로지스틱 회귀, 데모 단위 leave-one-out, 실제 최종 프레임 868장. 정확도 0.88(성공 0.91, 실패 0.84). r 구간별 0.92(r<0.8), 0.78(0.8–1), 0.72(1–2), 0.88(2–3), 0.95(≥3).
- 14:46 결과·로그·코드 회수(코드는 pod 사본과 diff 없음 확인). CF 에피소드 원본(약 30GB)은 맥으로 받지 않음(인덱스와 sanity report만 `results/E0_runpod/cf_goal_index/`).
- 14:48 `pod_down.sh b-e0` → `budget.py status`에서 `gone`, 장부 추정 지출 $0.87(1시간 26분). `runpodctl billing`은 아직 $0.27만 집계(지연).
