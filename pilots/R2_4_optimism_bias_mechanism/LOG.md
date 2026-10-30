# R2-4 작업 로그 (pilot `r2n4`)

| 시각 (UTC) | 작업 |
|---|---|
| 04:20~04:25 | PILOT_RULES, IDEAS_ROUND2 R2-4 절, NOVELTY_R1 3절(MiraBench, WorldOracle), 원문 에세이 3개 절(Rare Things and Memorization, Do Weak Activations Matter?, Does Scale Help or Hurt Superposition?) 읽음 |
| 04:25~05:07 | 설계(구간 균등 정렬 noise로 지지 일치, 미끄러짐 모드, 짝 교환·대조 설계), 로컬에서 코드(common/train/select_T/analyze/finetune/summarize/figures)와 PLAN.md 작성. `py_compile`만 실행 |
| 05:08 | pod `pai-r2n4` 생성. RTX A5000 재고 없음 → RTX 3090(qutw8j5mi8kon3, $0.50/h, 256 CPU). torch 2.4.1+cu124. rsync는 apt로, matplotlib은 pip로 설치 |
| 05:10 | 코드 push, 스모크 테스트 시작(N=5k, 400 step, seed 1개, 별도 출력 폴더) |
| 05:11 | 스모크 통과(학습 2초, 분석 모델당 0.4초, 미세조정·요약·그림까지). 미학습 스모크 모델에서 학습 전 null의 Jaccard 비율이 약 5.6배로 나옴 → readout 이득 공유 교란. 판정 기준은 그대로 두고 이득을 뺀 활성 변화 겹침과 학습 전·p 추세 비교를 보조 지표로 PLAN 10장에 추가(본 실행 전) |
| 05:12~05:14 | 보정 실행(p=0.3 std × T 4개 × seed 3, 123초). 규칙대로 T=10k → PLAN 8장 기록. 원격 실행 명령의 ssh 세션이 한 번 멈춤(백그라운드로 넘어감) → 이후 `( setsid nohup ... & )`와 로컬 `< /dev/null`로 실행 |
| 05:15 | 본 실행 시작(`run_main.sh`): 24개 모델 학습 → 분석 → 미세조정, 보조 N=20k/300k 학습은 병렬 |
| 05:15~05:18 | 본 실행 완료: 24개 모델 학습 29초, 분석 약 45초, 미세조정, 보조 N=20k/300k 학습(각 약 35초), 요약·그림. `criteria.json` 판정: S1 실패(비율 17.8배는 통과, 성공 MSE 차이 68%로 실패), S2 실패(p=0.01 D_f 13.4%p), S3 실패 → 사전등록 판정 **없음** |
| 05:20~05:23 | 사후 분석 1(PLAN에 없던 것): N=20k/300k 모델에도 기전 분석 실행, `posthoc.py`로 실패 정보 집중도(n50)·층·영역과 낙관 편향의 관계, 결정 경계 r*, 학습 중 낙관 편향 변화 정리 |
| 05:24 | 사후 분석 2: 학습 길이 4배(T=40k, N=100k, std·rw, seed 3) 학습과 기전 분석. 낙관 편향이 학습 지연(일시적)인지 확인 |
| 05:25 | 그림 3의 범례가 막대를 가리고 그림 4의 축 이름이 잘려 수정 |
| 05:27 | 결과·로그 회수(18 MB, 체크포인트 제외). pod와 로컬 code/의 md5 일치 확인 |
| 05:28 | `pod_down.sh r2n4` → `budget.py status`에서 r2n4 **gone**, 비용 $0.17 |
| 05:30~05:40 | REPORT.md 작성. 하위 에이전트 Write 도구가 보고서 파일을 막아 PILOT_RULES 6장대로 Bash heredoc으로 저장 |
