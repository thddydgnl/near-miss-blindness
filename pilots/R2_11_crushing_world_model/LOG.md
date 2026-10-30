# R2-11 LOG

| 시각(UTC) | 내용 |
|---|---|
| 07:41–07:55 | 문서 읽기: PILOT_RULES, IDEAS_ROUND2 R2-11, R2-2 REPORT·LOG·PLAN·code, 에세이 "What Neurons Do Under Pressure"·"What Dictionaries Learn", 기억 파일럿 SLI REPORT·PLAN(교훈: 라벨 순열 null 오염 → 회전·학습 전 null) |
| 07:55–07:58 | R2-2 code를 복사(R2-2 폴더는 건드리지 않음). `env.py`에 상자 회전 추가(원판-회전 사각형 접촉, 접촉점 충격량, 관성, 각속도 마찰, 회전 범위 벽 처리, 회전 렌더). `gen_data.py`(GT 전체 기록), `model.py`(폭 설정), `train.py`(시드별 lockstep) 작성. 로컬은 py_compile만 함 |
| 07:58 | `pod_up.sh r2n11`: RTX 4090(secure, $0.89/h) `mzw5fr2pyiq3y6`. 96 CPU, 251 GB RAM(컨테이너 한도 31 GB), torch 2.4.1 |
| 08:00 | **오류**: torch.hub(GitHub)·HF·files.pythonhosted 연결 시간 초과. 일부 호스트의 IPv6 경로 문제로 판단, `/etc/gai.conf`에 IPv4 우선을 넣고 재시도해 DINOv2 다운로드 성공. matplotlib은 08:44에 재시도해 설치됨 |
| 08:03 | 데이터 smoke(5%): 재실행 오차 0.0, 회전 렌더 확인(그림 strip을 받아 눈으로 확인) |
| 08:05 | **오류**: 전체 데이터 생성이 train 통계 계산 중 OOM-kill. 컨테이너 메모리 한도 31 GB인데 Z.float() 복사본이 여러 개 생김. 통계를 100 에피소드 단위 chunk로 계산하게 고치고 재생성(90 s, train 5.0 GB) |
| 08:08–08:20 | 학습 smoke(smoke 데이터 600 step, 7개 설정 모두 학습됨). timing: 두 프로세스 병렬 batch 128은 232 ms/it, batch 256은 360 ms/it → batch 256, 6,000 step으로 결정 |
| 08:22 | PLAN.md 사전등록 |
| 08:23 | 본 학습 시작(`run_main.sh`, 시드 0·1 두 프로세스, 각 7개 모델 lockstep) |
| 08:25–08:45 | 학습 중 `analyze.py`·`cem.py`·`judge.py` 작성 |
| 08:28–08:38 | 분석 smoke(smoke 체크포인트, 5% 데이터). 코드 경로는 통과했지만 데이터가 작아 값은 무의미. OMP 루프가 느려(모델당 3~4분) 벡터화함. 전체 규모 검증 실행은 학습 프로세스와 GPU 메모리가 겹쳐 OOM → 학습 뒤로 미룸 |
| 08:46 | PLAN 10.1 수정(판정 지표 계산 전): 혼동 지표를 상호작용 없는 표본 + 자기 혁신 통제(v2)로 바꾸고, 의미 행렬 A–S·R–C를 0.5로 조정. 원래 정의(v1)는 보조 보고 |
| 08:47 | 학습 종료 후 분석·CEM·판정이 자동 시작되게 연결(`wait_then_after.sh` → `run_after.sh`) |
| 09:00 | 학습 종료(36.6분). 최종 ID loss/copy(시드 0/1): w256 0.485/0.479, mlp64 0.503/0.509, mlp16 0.519/0.528, mlp8 0.520/0.528, res64 0.704/0.688, res16 0.879/0.874, res8 0.925/0.928. 분석과 CEM이 자동으로 병렬 시작 |
| 09:01 | 공유 probe: oracle 혁신 R²가 A 0.58, R 0.23만 0.2를 넘음(B 0.12, C 0.195, W 0.10, S 0.03). 사전등록 규칙대로 해석 가능 변수 = {A, R} |
| 09:05 | **오류**: 사후 차분 probe를 본 분석과 동시에 돌리다 GPU OOM → 본 분석 종료 뒤 재실행해 정상 완료 |
| 09:06 | CEM 완료(과제 61개: 밀기 29, 자유 32. 밀기 과제는 32개를 다 찾지 못함). 분석 완료(7.3분), judge 실행 → 사전등록 판정 **없음** |
| 09:13 | 사후 미세/거친 AUROC 분석, judge 재실행(그림 3 범례 위치만 수정) |
| 09:16 | 사후 양성 대조(혼동 지표 검정력) 완료 |
| 09:16 | 결과·로그 pull(체크포인트·데이터 제외). pod 코드와 로컬 `code/` 비교: pod에서 만든 `wait_then_after.sh`만 달라서 로컬로 복사함 |
| 09:16 | **사고(경미)**: 비교용 pull을 공유 scratchpad의 기존 폴더 `podcode/`(다른 파일럿이 쓰던 폴더)에 해 버림. 내 코드와 바이트가 같은 파일 12개만 지워 되돌림. 다른 파일럿의 같은 이름 파일을 덮어썼는지는 확인할 수 없음(그 폴더 파일은 모두 `*12.py` 등 다른 이름) |
| 09:17 | `pod_down.sh r2n11` → `budget.py status`에서 **gone** 확인, 비용 **$1.20**(1.31 h) |
| 09:22 | REPORT.md 작성. Write 도구가 하위 에이전트의 보고서 파일 쓰기를 막아 규칙대로 Bash heredoc으로 저장 |
