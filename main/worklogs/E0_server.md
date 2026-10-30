# B E0 서버 작업 기록 (M1·M3, 대학 GPU 서버 GPU 1)

- 작업자: B E0 서버 담당. 시각은 UTC(2026-10-21).
- 서버: `kiis-mvf-gpu`, RTX A5000 24GB × 2 중 **GPU 1만**(`CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES=1`). GPU 0은 A 작업자가 사용 중.
- 작업 폴더 `~/pmx/B/`(code, ext, logs, feat, results, ckpt, data_links), 환경 `~/pmx/envs/b-dinowm`, `~/pmx/envs/b-dp`, 기본 Python `~/pmx/envs/b-python`(A와 분리).
- 코드 원본은 맥북 `B_near_miss_blindness/main/code/server/`이고 rsync로 `~/pmx/B/code/`에 올려 실행했다.

| 시각 | 작업 |
|---|---|
| 13:08 | 서버 점검: 디스크 여유 317GB, GPU 둘 다 비어 있음. `~/pmx/B/{ext,code,logs,feat,results,ckpt,tmp}` 생성 |
| 13:15 | `dino_wm` clone(0a9492fa, R2-12와 같은 commit), `diffusion_policy` clone(5ba07ac6) |
| 13:16 | DP 데이터 서버(diffusion-policy.cs.columbia.edu)가 모든 경로에 404. HF에서 공식 zarr의 미러 `lerobot-raw/pusht_raw`(rev f93fc579)를 찾음: 206 episode, 25,650 step, 공식 구조와 같음 |
| 13:19~13:22 | `setup_envs.sh`: uv로 Python 3.11.17, torch 2.4.1+cu124, gym 0.23.1, pymunk 6.2.1, diffusers 0.30.3, zarr 2.18.3, numba 0.60, matplotlib 3.8.4(DP keypoint manager의 `cm.get_cmap` 때문에 3.9 미만) 등. 두 환경 모두 같은 패키지 집합. DINOv2 hub는 `~/pmx/torch`에 이미 있었음 |
| 13:21 | `~/pmx/B/data_links/pusht_noise` → `~/pmx/data/pusht_noise`(읽기 전용). DP 시연을 `~/pmx/data/dp_pusht/`(새 빈 폴더, 36MB)로 내려받음 |
| 13:23 | `dp_conventions.py`(시뮬레이터·numpy만): C1~C6 모두 통과(보고서 3장) |
| 13:24 | M1 재현 시작(tmux `B-E0-m1repro`, `run_m1_repro.sh`): 스모크 통과(빠른 재생 = env.rollout, 상태·이미지 차이 0) |
| 13:25 | 시뮬레이터 보정 H5·H10이 R2-12와 비트 단위로 같음(격자 최대 차이 0.0, 정책 모수·확인 수치 동일). 판별기 학습 집합도 같음(구간 비율, 사용한 train 궤적 목록 동일) |
| 13:26~13:30 | DP 학습 스모크: 3 epoch, 65.8M 파라미터, epoch당 42 batch. 재개 시험에서 RNG 상태 복원 오류(`map_location`이 RNG 텐서를 GPU로 옮김) → `.cpu()`로 고침, 재개 확인 |
| 13:31 | DP rollout runner 스모크(4 epoch 스냅샷, 20 episode): 동작 확인(약한 정책이라 성공 0%, far 95%) |
| 13:32 | **DP 학습 시작**(tmux `B-E0-dp`, 로그 `~/pmx/B/logs/B-E0-dp.log`, seed 42·43·44 동시, 1000 epoch, EMA 스냅샷 23개) |
| 13:35 | 판별기 완료(340초): (a) held-out 0.9552, (b) τ 0.2125 등 R2-12와 같음. main 집합 400 episode 169초(GPU 공유 상태) |
| 13:35 | M3 시작(tmux `B-E0-m3`, `run_m3.sh`) |
| 13:38 | `cf_grid_pilot.py`(시뮬레이터만): 반사실 severity를 표적 r*에 맞추는 이분 탐색이 정밀하게 동작함을 확인 |
| 13:41 | `dp_imagine.py` 스모크: DP episode의 마지막 5H step 창을 DINO-WM으로 상상(H 3·5·7·10), 재생 오차 최대 8e-11 px |
| 13:46 | M1 재현 끝(main 169초, H10 281초, trainsplit 187초, 분석·사후 분석 38초). 맥북에서 비교: 시뮬레이터·(b)·(b2) 지표가 소수 넷째 자리까지 같음, 상상 특징 차이 d_vis 최대 0.0015. 판별기 (a)만 운영점이 달라짐(실제 프레임 P(성공 \| near) 0.146 → 0.197) |
| 13:48~14:17 | 진단: `analyze_robj_e0.py`(물체 자세만 정의로 R2-12 집합 재분류), `judges_obj_e0.py`(r_obj 라벨 판별기, held-out 0.962), `judge_seeds_e0.py`(판별기 (a) seed 5회: held-out 0.953~0.958, AUROC 안정, 0.5 문턱 운영점은 흔들림) |
| 14:00 | M3 p=0.03 1× seed 0이 R2-9와 같은 값(mean over 0.144, π6 +0.441, π5 +0.041, r* 1.388, n50 558) |
| 14:07 | 조정자 점검 메시지: (1) DP checkpoint와 특징 캐시 디스크 정리, (2) DP 로그의 `val nan` 확인 요청 |
| 14:10 | `val nan` 원인: 검증은 25 epoch마다와 스냅샷 epoch에만 돌고, 나머지 epoch에서 출력 형식이 nan을 찍었음. 계산된 검증 손실 26개/seed는 모두 유한값. 출력을 `skip`으로 고침(재시작 시 적용) |
| 14:11 | dp_smoke 삭제(0.75GB). 가장 이른 스냅샷 2·4·6 삭제 |
| 14:15 | DINO-WM 미세조정 벤치(GPU 혼잡 상태): batch 32에 1.94초/step, 최대 메모리 5.6GB |
| 14:26 | M3 끝(3,025초): p=0.03 1× seed 3개 평균이 R2-9와 비트 단위로 같음(16.9%p, π6 +45.9, π5 +9.8, 4.68배, r = 0.954, r* 1.4955, n50 539). p=0.1 0.084, p=0.3 0.032 |
| 14:26 | seed 42 스냅샷의 시뮬레이터 평가(보정 seed 200000~200049, 50 episode) 결과로 8단계 결정: epoch 20, 25, 30, 40, 50, 80, 220, 1000. 나머지 EMA 스냅샷 삭제와 이후 생기는 300·400·550·750 자동 삭제(tmux `B-E0-dpprune`) |
| 14:27 | 특징 캐시 정리: M1 `clf.pt`, `eval_*.pt`(8.2GB), M3 bank 3개(4.8GB) 삭제(모두 결정론적으로 다시 만들 수 있음). ckpt/dp 8.1GB, feat 0.33GB, 서버 여유 294GB |
| 14:27 | seed 43·44의 7단계 시뮬레이터 평가 시작. 표준 목표 판별기 (a_std) 가능성 점검용 rollout 시작(판별기 학습 seed 300000~, 시연 seed 900000~; E2 평가 seed 100000~은 쓰지 않음) |
| 14:41 | **사고 발견**: tmux 세션을 인라인 명령(`tmux new-session -d "export ...; python ..."`)으로 띄우면 내 셸의 export(`CUDA_VISIBLE_DEVICES=1`, `TORCH_HOME`)가 전해지지 않는다(tmux 서버 전역 환경에 없음). 이렇게 띄운 B 작업 셋이 기본 장치 = **GPU 0(A의 GPU)**에서 돌았다: `judge_seeds_e0.py`(13:48~14:17), `judges_obj_e0.py`(13:49~13:56), a_std 점검(14:40~14:42). 마지막 것은 `/proc/<pid>/fd`에서 `/dev/nvidia0` 사용과 환경 변수 부재로 확인했다. 앞의 둘은 같은 방식으로 띄웠으므로 GPU 0으로 본다(이미 끝나 직접 확인 불가). 스크립트(`run_*.sh`, `snapeval_list.sh`, `rollouts_e0_astd.sh`)로 띄운 작업은 모두 benv.sh를 읽어 GPU 1이었음을 확인했다 |
| 14:42 | a_std 작업 즉시 종료. 그 작업이 TORCH_HOME 없이 DINOv2 hub를 `~/.cache/torch/hub/`(facebookresearch_dinov2_main, checkpoints/dinov2_vits14_pretrain.pth 84MB, trusted_list)에 받았다. `~/pmx` 밖이라 규칙대로 지우지 않았다(조정자 보고) |
| 14:43 | 재발 방지: `bpaths.py`와 M3 스크립트 3개에 가드 추가(`CUDA_VISIBLE_DEVICES=1`, `CUDA_DEVICE_ORDER=PCI_BUS_ID`가 아니면 실행 거부, TORCH_HOME·HF_HOME 기본값을 `~/pmx` 안으로). tmux 실행은 모두 benv.sh를 읽는 스크립트(`bjob.sh`, `run_astd.sh`)로만 한다. 남은 B 프로세스 전부가 GPU 1임을 `/proc`로 확인 |
| 14:44 | a_std 점검을 GPU 1에서 다시 시작(`run_astd.sh`, tmux `B-E0-astd`) |
| 14:45 | 조정자에게 GPU 0 사고 알림(SendMessage) |
| 14:50 | a_std 점검 결과(GPU 1): 표준 목표 판별기 held-out 0.975, DP 창(seed 43 단계 25·30·50, 300 episode)에서 성공 대 near AUROC 실제 0.998 → 상상 0.970(H3) ~ 0.887(H10), near 오판 0.12 → 0.22~0.33 |
| 14:50 | seed 43·44 단계 평가 완료(`dp_snapeval_summary.json`). 단계별 near 비중 0.02~0.36 |
| 14:52 | 진단용 특징 파일 삭제(`demo_imag.pt` 0.72GB 등), feat 0.29GB |
| 14:53 | 쓰지 않은 `run_dp_snapeval.sh` 삭제. 맥북 코드와 서버 코드 md5 일치 확인 |
| 15:26 | 서버 쪽 대기 루프가 ssh 세션을 많이 잡아 multiplexing이 거부됨 → 내 대기 루프만 정리(A의 루프는 건드리지 않음) |
| 15:34~15:35 | **DP 학습 완료**(세 seed, 1000 epoch, seed당 7,350~7,409초, 재시도 없음). 검증 손실 55회/seed 모두 유한값. ckpt/dp 8.9GiB(단계 EMA 8개 + latest.pt) |
| 15:37 | GPU 1 처리량 측정 시작(`post_dp.sh` → `bench_uncontended.sh`, tmux `B-E0-post`): epoch 1000 단계의 시뮬레이터 평가 포함 |
| 15:37~15:46 | GPU 1 단독 처리량: M1 평가 집합 H5 3,600 궤적 61초, H10 2,700 궤적 79초, DP rollout 0.4초/episode, DP 창 상상 100개 4.8초, M3 bank 105초·학습 71초·평가 35초(E0 결과와 정책별 차이 0.0), DINO-WM 미세조정 0.345초/step. epoch 1000 단계 평가(seed당 50 episode): r_obj 성공 0.76~0.80, near 0.04~0.06 |
| 15:46 | 단계 요약·그림(`dp/fig_dp_stages.png`). B tmux 세션 모두 종료. 최종 디스크: ckpt/dp 8.9GiB(9.5GB), ckpt/m3 36MB, feat 0.29GB, results 0.11GB, 서버 여유 293GB |
| 15:50~ | 결과를 맥북 `results/E0_server/`로 동기화(서버 로그 포함), 보고서 작성 |
