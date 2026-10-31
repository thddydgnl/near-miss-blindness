# TMLR 원고 구성안 — Near-Miss Blindness

- 작성: 2026-10-31. 근거: `main/DESIGN_main.md` v1~§11, `main/reports/FINAL_B.md`, 단계 보고 7개, JKAIA 포스터 최종본(10-25), `paper/README.md` §2·§4
- 원고: [`main.tex`](main.tex). 이 문서는 구성의 이유와 각 절의 근거를 적는다.

## 1. 증거 진단

| 주장 후보 | 근거 | 강도 | 원고에서의 처리 |
|---|---|---|---|
| latent 상상은 결과 경계의 해상도를 잃는다 | M1 H1b 성립(1차 가설), H4b 성립, trainsplit 반복, horizon 기울기 | 강함. 공개 모델 | 핵심 주장 1. 단, "near-miss 특이적"은 원 정의 r, H ≤ 5로 한정 |
| 공개 모델의 오류는 낙관이 아니라 양방향 흐림이다 | M1 H1a 불성립(0.147), 성공 판정 0.862 → 0.710 | 강함 | 주장 1의 일부. "월드모델은 낙관적"이라고 쓰지 않음 |
| 성공 편중 데이터가 흐림을 낙관으로 바꾼다 | M3 H4a·H1a 성립(용량-반응), M1 미세조정은 부분 결과 | 인과는 통제 모델에서만 | 핵심 주장 2. M1 부분 결과와 한계 병기 |
| near-miss 오판의 주된 원천은 상상이다 | H3 M2·M3 성립(탐색), M1 불성립 | 중간. M2에 판정기 단서 | 주장 3. 단서를 본문과 그림 캡션에 |
| 정보 손실(M1) 대 분포 이동(M3) | H4b M1 성립·M3 불성립 | 중간 | 주장 3과 같은 절에서 "오류가 들어오는 자리"로 |
| 예측기 용량이 경계 해상도를 크게 바꾼다 | M3 sweep(사전등록 밖) | 서술 | "not pre-registered" 표시 |
| 정책 과대 추정이 near-miss 비중을 따른다 | M3 H2a 성립, M1 불성립(편향 작음), 역전 없음 | 모델 하나 | 주장 4. 순위 역전 없음과 M1의 작은 편향을 같이 |
| 처방 | H5a 0/3(부정), H5b 성립(작음) | 부정·약함 | 부정 결과로. 경계 겨냥 자료를 처방으로 제안하지 않음 |
| 비디오 월드모델 | M2 no-go(사전 규칙) | 부정 | 제목과 범위를 latent로 한정한 근거. 같은 측정 절에 대비로 |

- 1차 가설군(H1a·H1b·H2a)은 각각 모델 하나에서만 성립해 "모델 2개 이상" 기준을 못 넘는다. 이것을 초록, 서론, 표 2, 한계에 모두 밝힌다.
- 주장 두 개가 서로 다른 모델에 기댄다(정보 손실은 M1, 낙관과 정책 법칙은 M3). 숨기지 않고 한계 절 첫 항목으로 쓴다.
- TMLR 심사 기준은 "주장이 정확하고 설득력 있는 증거로 뒷받침되는가"와 "일부 독자에게 흥미로운가"다(2026-10-31 공식 페이지 확인). 그래서 주장의 범위를 증거에 맞추고, 사전등록과 부정 결과를 강점으로 쓴다.

## 2. 구성 원칙

1. **실험 단계(E1~E5)가 아니라 주장 순서로 쓴다.** 절마다 주장 하나, 그 주장의 사전등록 검정, 단서를 둔다. 독자는 "무엇이 → 어디서 → 왜 → 그래서 → 고칠 수 있나"의 순서로 읽는다.
2. **M2는 따로 떼지 않고 같은 측정 절(4.3)에 대비로 넣는다.** 같은 프로토콜이 비디오 월드모델에서는 경계에 몰리지 않는 흐림을 보인다는 것이 그 자체로 정보다. 사전등록이 H1을 세 모델 모두에 걸었으므로 그대로 보고한다. 주장은 latent 모델로 한정하고 제목에도 "Latent"를 넣는다.
3. **결과마다 표시를 단다**: pre-registered primary / pre-registered exploratory / not pre-registered(사후·서술). 표 2가 전체 판정을 한 번에 보여 준다.
4. **본문은 참고문헌 앞까지 12쪽 이하**로 맞춘다(TMLR: 12쪽 이하는 심사 2주, 넘으면 4주). 세부와 모든 칸은 부록으로 보낸다.
5. **이중맹검**: 저자, 소속, 서버·학교 이름, KAIA 포스터, 실명 GitHub 저장소를 원고에 쓰지 않는다. 코드는 익명 보충 자료로 낸다.
6. **수치**: 본문의 모든 수치는 `build/verify_numbers.py`가 결과 JSON과 대조한다. 그림은 `build/make_figs.py`가 결과 JSON만 읽어 그린다(새 계산 없음).

## 3. 목차

| 절 | 제목 | 내용 | 그림·표 | 분량 |
|---|---|---|---|---|
| — | Abstract | 질문, 방법, M1·M3 핵심 수치, 처방, M2 no-go와 범위 | — | 0.4쪽 |
| 1 | Introduction | 정책 평가용 월드모델, 성공 환각 보고, "어떤 실패가 오판되나", 측정 방법, 발견 4개, 범위와 부정 결과 | 그림 1: 측정 절차 + 두 가지 오류 모양(낙관, 양방향 흐림) | 1.5쪽 |
| 2 | Related Work | 월드모델 정책 평가, 성공 환각과 반사실 실패, 실패 감지·판정기, 순위 지표, 사전등록 | — | 0.7쪽 |
| 3 | Measuring Boundary Resolution | 3.1 실패 거리 r과 상상 판정 · 3.2 짝 판정(지각 바닥, 상상 몫) · 3.3 경계 해상도와 정보 상한 · 3.4 정책 수준 양과 FGRE · 3.5 경계 밀집 반사실 · 3.6 모델·판정기·정책 · 3.7 사전등록과 통계 | 표 1: 모델·판정기·평가 집합. 표 2: 사전등록 판정 전체 | 2.3쪽 |
| 4 | Imagination Loses Resolution at the Boundary | 4.1 M1(H1b 성립, 양방향 흐림, 특이성의 범위) · 4.2 M3(낙관 모양, 보수적 집계 규칙) · 4.3 M2(경계에 몰리지 않는 흐림, no-go) | 그림 2: 세 모델의 실패 거리 곡선(M3는 p별). 그림 3: M1 AUROC와 양방향 이동 대 상상 길이 | 2.0쪽 |
| 5 | Where the Error Enters | 5.1 판정기 대 상상(H3, VLM 자체 낙관) · 5.2 정보 손실 대 분포 이동(H4b) · 5.3 예측기 용량(사전등록 밖) | 그림 4: 2×2 분해. 용량 sweep 표는 부록 F | 1.2쪽 |
| 6 | Success-Skewed Data Shift Imagination toward Optimism | M3 용량-반응(H4a), M1 미세조정 부분 결과 | 그림 2(c)를 함께 쓴다(p별 곡선). NM·SM 값은 본문 | 0.6쪽 |
| 7 | Consequences for Policy Evaluation | M3 정책 법칙(H2a), 큰 상상 차이지만 순서 유지(H2b), 표준 지표(H2c), M1의 작은 편향, FGRE 대 MMRV | 그림 6: M3 정책 6개, M1 DP 24개 | 1.2쪽 |
| 8 | Remedies | 노출량 맞춘 데이터 처방(H5a 부정), 재가중의 큰 효과(탐색), 경계 가중 손실, MC dropout 할인(H5b, 작음) | 표 4: 처방별 NM·SM | 0.8쪽 |
| 9 | Discussion and Limitations | 평가 실무에 주는 함의, 한계(모델 수와 주장의 분리, 시뮬레이션, 판정기 의존, M3 구조, 짧은 미세조정, E6 미시험) | — | 0.9쪽 |
| 10 | Conclusion | | — | 0.2쪽 |
| — | Broader Impact Statement | 월드모델 평가의 과신 위험 | — | 0.1쪽 |
| A | Pre-registration Record | 설계 문서 버전별 시각·sha256, 결과 전·후에 정한 것, 편차 목록 | 표 | 부록 |
| B | All Pre-registered Tests | FINAL_B §2 표 전체(구간 포함), 다중 비교 | 표 | 부록 |
| C | M1 Details | DINO-WM PushT, r 정의 두 가지, 반사실 생성, 분할, 판정기, DP 학습·단계 선택, 모드 O, VLM 절차, 미세조정, MC dropout | 표 | 부록 |
| D | M2 Details | Cosmos Policy, 과제와 τ, 정착 상태 r, 교란 계열, 질의 규약, 판정기, CureWM 재현 | 표, 그림(실제 대 상상 프레임) | 부록 |
| E | M3 Details | 장면, 인코더, 예측기, 학습, 정책 π1~π6 | 표 | 부록 |
| F | Additional Results | M1 모든 칸, 판정기 (b), trainsplit, M2 과제별·k별·특징 판별기, 손실 단계 몫(E4c), 순위 지표 표 | 표 | 부록 |
| G | Compute | RunPod A40 비용, 서버 GPU 시간 | 표 | 부록 |
| H | Reproducibility | 코드 구성, 상류 저장소 커밋, seed, 수치 재계산 | — | 부록 |

## 4. 쓰지 않는 주장 (FINAL_B §3.3, DESIGN_main §8·§10)

- "월드모델은 낙관적이다"(일반 명제)
- 비디오 월드모델(Cosmos Policy)에서 near-miss blindness가 성립한다. M2의 과제별 성립(t6 등)이나 문턱 근처 값을 근거로 쓰는 것
- 성공률이 같은 정책의 순위가 상상에서 뒤집힌다
- 경계 겨냥 반사실 자료가 처방이다
- checkpoint 선택에서 regret이 생긴다(E6 없음)
- M3 용량 sweep, 재가중 효과, E4c를 사전등록 결과처럼 쓰는 것
- "최초 발견"

## 5. 포스터에서 뺐다가 되살리는 것 (paper/README §4)

- World-VLA-Loop 인용, 통계 항목(95% CI, Holm 보정), E6 생략을 한계 절에

## 6. FINAL_B §9 집필 할 일의 처리

| # | 할 일 | 처리 |
|---|---|---|
| 1 | H3 그림에 M2 칸 | 그림 4에 `h3_m2.json` 값으로 넣음 |
| 2 | 좁힌 범위로 서론·관련 연구, 10-21 이후 관련 논문 | 2절. 참고문헌 59항목을 출판처까지 확인했다([`REFERENCES_CHECK.md`](REFERENCES_CHECK.md)). 가장 가까운 새 연구 WorldFailBench(10-24)와의 차이를 2절에 적었다 |
| 3 | 그림을 논문 양식으로 | `build/make_figs.py`(JSON만 읽음) |
| 4 | 표 | 표 1~4, 부록 표 |
| 5 | 부정 결과를 본문에 | 4.3, 7, 8, 9절과 표 2 |
| 6 | 사전등록 부록 | 부록 A |
| 7 | 다중 비교 단락 | 3.7절, 부록 B |
| 8 | 재현 부록 | 부록 H |
| 9 | 비용 | 부록 G, pod 장부 기준 $20.27 |
| 10 | 수치 정의 통일(D4, D5) | D4: "상상이 더한 과대 추정 o^imag"로 명시. D5: r50은 로지스틱 맞춤 값(E1 정의)으로 통일 |
| 11 | 투고처와 일정 | TMLR, 본문 12쪽 이하 |

## 7. 독립 검토 뒤 바뀐 점 (2026-10-31)

- 검토 보고서: [`REVIEW_2026-10-31.md`](REVIEW_2026-10-31.md). 필수·권장 항목을 모두 반영했다(요약은 `../README.md` 6.3절).
- 구성 변경: 옛 그림 5(M3 p별 곡선과 NM·SM 막대)를 그림 2(c)에 합쳤고, 용량 sweep 표를 부록 F로 옮겼다. 6절 제목을 "Success-Skewed Data Shift Imagination toward Optimism"으로 바꿨다(M3 안에 흐림 기준 조건이 없어 "blur를 optimism으로 바꾼다"는 근거가 약함).
- 본문은 참고문헌 앞까지 12쪽 안이다(심사 2주 기준).

## 8. 2차 수정 (2026-10-31): 사전등록 서술을 낮춤

- 사용자 질문("꼭 사전등록을 써야 하나")에 대한 판단: TMLR 필수가 아니고 ML 논문에 드물다. 완전히 빼면 문턱값과 M2 제외가 사후 선택으로 보이므로, "본 실험 전에 고정한 분석 계획"으로 낮춰 부르고 본문에서는 빼며 부록 A에 기록을 둔다.
- 바뀐 구성: 3.7절 "Analysis plan and statistics"(4문장), 표 2 "The tests, with thresholds fixed before the main runs"(그룹명 Primary tests / Other tests), 부록 A "Analysis Plan Record"(무엇을 안 상태였나, 보수적 집계 규칙 포함), 부록 B "All Planned Tests". 결과 절은 효과 크기로 서술하고 H 라벨은 괄호 참조로만 쓴다.
- 함께: 서론 기여 단락, bullet 숫자 축소, reference judge, 6절 제목 변경, 용량 결과를 6절 끝으로.

## 9. E7 반영 (2026-11-01): 두 번째 공개 DINO-WM 과제 PointMaze

- 근거: [`../../main/reports/E7_runpod.md`](../../main/reports/E7_runpod.md) §5(원고가 쓸 수치와 JSON 경로), 설계 §12. 판정 칸(val, H = 5, r, 판정기 (a)): H1b(E7) 불성립(near 0.197 ✓, far 0.076 ✗, 기울기 −0.033 ✓), H4b(E7) 성립(0.980 대 probe 0.778).
- 자리: E7은 §12.0대로 모델 수에 넣지 않는 복제 단계다. 그래서 M4를 만들지 않고 "M1 on two tasks"로 부른다. 4.1절 제목을 "A public latent world model (M1), on two tasks"로 바꾸고 끝에 "A second public task" 단락을 두었다. 표 2에는 집계 밖 블록(H1b·H4b 두 행, Overall "not counted")으로, 부록 B에는 "(replication, not counted)" 두 행으로 넣었다. 부록 A의 버전 표에 §12 행(해시 `3273f801e657`, "after all verdicts and a first draft; before any PointMaze evaluation set was imagined")과 편차 (15)를 더했다. 새 부록 D에 설정·결과·편차와 칸별 표(val·train × H 4개).
- 그림: 그림 3을 3패널로 바꿨다. (a) PushT 해상도, (b) PointMaze 해상도(같은 계열; probe가 상상 판정과 겹침), (c) PushT 양방향. 본문의 그림 3b 참조는 (c)로 옮겼다.
- 서술 규칙: "near-miss 특이적" 손실은 PushT r·H ≤ 5에 한정한다(v2.1 §6). 두 과제를 함께 설명하는 문장은 "손실은 경계에서 시작해 horizon과 함께 바깥으로 번진다"(Δr10 PushT 0.15→0.47, PointMaze 0.50→2.19; near 손실이 far 손실의 2.6배). 초록은 "concentrated at the boundary without being confined to it". H4b는 "공개 모델의 두 과제 모두"로 쓰되 모델 수는 늘리지 않는다. 9절 한계에 "strict near-miss specificity on PushT only"를 넣었다.
- 분량: 본문이 12쪽을 9행 넘어 중복 문장을 줄였다(README 6.3 v3 항목). 참고문헌이 12쪽 중간에서 시작한다.
