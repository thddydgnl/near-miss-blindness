# 본 실험 작업 규칙 (A·B 공통, 작업자는 시작 전에 끝까지 읽는다)

프로젝트 루트(공백 포함, 항상 따옴표): `/Users/song-yonghwi/Documents/physical ai - Neural Networks`

## 0. 역할과 기준 문서
- 조정자(메인 세션)가 단계를 나눠 맡기고, 작업자는 맡은 단계를 끝까지 수행한 뒤 보고한다.
- 기준 문서
  - A: `A_dithered_world_models/EXPERIMENT_DESIGN.md`
  - B: `B_near_miss_blindness/EXPERIMENT_DESIGN.md`
- 가설, 지표, 판정 문턱, 과제 목록, 주장은 바꾸지 않는다.
  - 설계대로 시험할 수 없다고 판단되면(핵심 구성요소가 안 돌아감, 가설을 쓰인 대로 잴 수 없음 등) 거기서 멈춘다.
  - 선택지와 근거를 조정자에게 보고한다. **사용자에게 묻는 것은 조정자의 일이다.**
- 설계 문서에 이미 적힌 대체 경로는 써도 된다. 다만 LOG.md에 기록하고 보고에 적는다.
  - A: PointMaze 제외
  - B: M4 생략
  - 예외: 비용이 드는 대체 경로(A의 Meta-World를 RunPod로 옮기기)와 B의 M2 대체 경로(LIBERO판 통제 모델)는 조정자 승인을 받는다.
- 문서(LOG, 보고서)는 한국어로 쓴다. 코드와 주석은 영어로 쓴다.

## 1. 실행 장소
- **맥북**: 파일 편집, `python3 -m py_compile` 같은 문법 검사, 몇 초 안에 끝나는 순수 파이썬 점검, 결과 읽기만 한다.
  - 학습, 평가, 시뮬레이션, 렌더링, 모델·데이터 다운로드는 하지 않는다(맥북 과열).
- **대학 GPU 서버** (`ssh kiis-mvf-gpu`, 키 인증, BatchMode)
  - 서버에서 도는 것: A 전체, B의 M1·M3.
  - 작업 폴더: `~/pmx/A`, `~/pmx/B`
  - 환경: `~/pmx/envs/<이름>`(uv venv). uv는 `~/.local/bin/uv`이고, 캐시는 기본값 `~/.cache/uv`를 그대로 쓴다(hardlink로 공간 절약).
  - 데이터: `~/pmx/data`
    - PushT: `~/pmx/data/pusht_noise`(imle-wm 데이터를 가리키는 심볼릭 링크, 읽기 전용)
    - DINO-WM 체크포인트와 Wall 메타: `~/pmx/data/dinowm/`
  - 캐시 경로: HF 캐시는 `HF_HOME=~/pmx/hf`, torch hub는 `TORCH_HOME=~/pmx/torch`
  - GPU 배정: **A는 GPU 0, B는 GPU 1**(`CUDA_VISIBLE_DEVICES`). 다른 쪽 GPU는 조정자가 바꿔 주기 전에는 쓰지 않는다.
  - 제약: sudo, Docker, conda가 없고 OpenGL·EGL 라이브러리도 없다. 시스템 라이브러리가 필요하면 `~/pmx/envs/micromamba`에 micromamba로 사용자 공간 설치를 한다.
  - **홈의 다른 폴더는 건드리지 않는다**: DriveWAM_Lightweight_Action_Readout, jev-wm, aawm, imle-wm, decision-compaction, kiis-mvf*, hf, alfworld_data. 읽기 전용 pusht_noise만 예외다. `~/pmx` 밖에서는 아무것도 지우지 않는다.
  - 디스크는 공유이고, 시작 시점에 약 319GB가 남아 있었다.
    - checkpoint는 필요한 것만 남긴다(EMA 가중치 위주). 대체된 중간 산출물은 `~/pmx` 안에서 정리한다.
    - 남은 공간이 60GB 아래로 내려가면 멈추고 보고한다.
  - ssh 호스트 키 확인을 끄지 않는다. 비밀번호를 다루지 않는다.
- **RunPod** (B의 M2, 그리고 승인된 경우에만 A의 Meta-World 대체 경로)
  - 도구는 `runpod_main/`(pod 접두어 `pmx`)의 `pod_up.sh`, `rsh.sh`, `sync.sh`, `pod_down.sh`, `budget.py`만 쓴다. 사용법은 `runpod_infra/PILOT_RULES.md` §2와 같고, 경로만 `runpod_main`으로 바꾼다.
  - 상한
    - 단계별 상한: `runpod_main/caps.json`
    - 전체: 감시 스크립트가 $55에서 모든 pod를 지운다(사용자 상한 $60). 단계 상한의 80% 안에서 끝내도록 계획한다.
  - **다른 pod(`nnc-*`, `wmc-*`, `pai-*`)는 절대 건드리지 않는다.**
  - pod는 생성 4시간 뒤 자동 종료된다(TERM_H). 더 긴 작업은 checkpoint와 재개를 설계하거나, 조정자에게 TERM_H 연장(최대 8)을 요청한다.
  - 결과를 받은 뒤 `pod_down.sh`로 지우고, `budget.py status`에서 `gone`을 확인한다.

## 2. 사전등록
- E1 이후의 본 실행 전에 조정자가 `<논문 폴더>/main/DESIGN_main.md`를 고정한다.
- E0 작업자는 설계 문서와 E0 결과로 그 초안의 재료를 만든다: 과제 목록, k 선택 결과, seed와 episode 범위, 판정기 문턱 등. 가설과 문턱은 고치지 않는다.
- 키우기 전에 작은 설정으로 스모크 테스트를 한다.
- 결과를 본 뒤 정한 분석은 사후 분석으로 표시한다.

## 3. 긴 작업과 점검
- 1시간 넘게 걸릴 작업은 `MAIN_STATUS.md`의 작업 표에 올린다: 이름, 장소, tmux 세션 또는 pod, 시작 시각, 예상 종료, 로그 경로.
- 조정자가 1시간마다 `runpod_main/check_all.sh`로 점검한다. 긴 작업을 기다리는 작업자도 1시간 이내 간격으로 로그를 보고 오류를 고친다.
- 서버의 긴 작업은 tmux 세션(`A-<단계>-<설명>`, `B-<단계>-<설명>`)에서 돌린다. 로그는 `~/pmx/<A|B>/logs/<이름>.log`에 쓴다.
- runner는 다시 시작할 수 있게 만든다: 작업 목록이 있고, 작업마다 결과 파일을 쓰며, 이미 있으면 건너뛴다.
- 오류가 나면 고치고 이어서 돌린다. 오류와 수정은 LOG.md에 남긴다.

## 4. 산출물
- `<논문 폴더>/main/code/`: 실제로 돌린 코드 전부
- `<논문 폴더>/main/results/<단계>/`: JSON·CSV와 작은 PNG. checkpoint와 데이터는 맥북에 두지 않는다.
  - 예외(2026-10-21, 조정자): B의 M2 산출물(LIBERO DP checkpoint, 판정기 (a) 가중치)은 pod 사이 전달용으로만 맥북 임시 폴더 `/private/tmp/claude-501/-Users-song-yonghwi-Documents-physical-ai---Neural-Networks/d3639ab6-a0e1-42cf-ad6b-e084af659cc6/scratchpad/m2_artifacts`에 둘 수 있다. 프로젝트 폴더에는 두지 않는다.
- `<논문 폴더>/main/LOG.md`: 시간순 기록(장비, 오류와 수정)
- `<논문 폴더>/main/reports/<단계>.md`: 한국어 단계 보고
  - 무엇을 돌렸나
  - 수치
  - 사전등록 규칙에 따른 판정
  - 편차
  - 비용과 시간
- Write 도구가 실패하면 Bash heredoc으로 쓴다.

## 5. 조정자에게 보내는 마지막 메시지
- 단계 결과와 핵심 수치
- 편차, 그리고 결정이 필요한 것
- 아직 도는 작업: tmux 이름이나 pod, 로그 경로, 예상 종료
- RunPod pod 상태: 조정자가 남겨 두라고 하지 않았다면 반드시 `gone`
