# JKAIA 2026 포스터 요약문 독립 검토 (A·B) — 2026-10-25

검토 대상
- A: `A_dithered_world_models/paper/jkaia2026/main.pdf` (+ `main.tex`, `main_ko.tex`, `abstract_ko.txt`)
- B: `B_near_miss_blindness/paper/jkaia2026/main.pdf` (+ `main.tex`, `main_ko.tex`, `ABSTRACT_KO.md`)

대조 근거
- A: `main/DESIGN_main.md` v1~v5.3, `main/results/E1_tdmpc2/e1_tdmpc2_cells.csv`, `E1/e1_dinowm_cells.csv`, `E1/e1_dinowm_v52.json`, `E2_tdmpc2/*`, `E3_tdmpc2/e3_tdmpc2_final.json`, `E3_dinowm/*`, `E5_*/*`, `E7_tdmpc2/*`, `reports/E4_theory.md`, `reports/E7_tdmpc2.md`, `reports/posthoc_tdmpc2.md`
- B: `main/DESIGN_main.md` v1~§11, `main/reports/FINAL_B.md`, `E1_server.md`, `E1_m2.md`, `E2E3_server.md`, `E4E5_server.md`, `paper/build/verify_numbers.py` 실행(36개 수치 전부 JSON과 일치)
- 참고문헌: arXiv 7건(2411.04983, 2601.16163, 2610.09134, 2605.29360, 2602.02110, 2609.19441, 2610.05550)의 제목·제1저자를 arXiv 페이지에서 대조 → 모두 일치

표기: **[필수]** 제출 전에 고치는 것이 좋음 / **[권장]** 고치면 좋음 / **[선택]** 저자 판단

---

## 0. 공통 결론

- 구조: 두 편 모두 템플릿 구성(Abstract, Keywords, A. Problem Setup, B. Novelty, C. Algorithms, D. Experiments)을 지켰고 2쪽이다. 초록 줄 수는 A 11줄, B 14줄로 규정(10~20줄) 안이다.
- 수치: 두 편 모두 결과 파일에서 다시 계산한 값과 일치한다. 잘못 옮긴 숫자는 없다.
- 주장 범위: 사전등록 판정(성립·불성립)과 서술 범위(DESIGN_main 승인 사항)를 어기는 문장은 없다. 부정 결과(A: H2a·H2b·H4b·DMC·H3b DINO-WM, B: M2 no-go·1차 가설군 불성립·H5a)가 모두 들어 있다.
- 고칠 것은 대부분 **"숫자가 같은데 뜻이 다른 표현"과 "정의 없이 쓴 용어", "범위가 모호한 문장"**이다. 아래 항목별로 교체 문장을 적었다.

### 두 편에 걸친 공통 항목

| # | 등급 | 내용 | 제안 |
|---|---|---|---|
| C1 | 권장 | **사전등록 표현 불일치.** A는 "Internal pre-registration … frozen in timestamped documents"라고 쓰고, B는 "pre-registered"·"hashed amendments"만 쓴다. 같은 저자가 같은 학회에 내는 두 편이라 심사자가 B를 OSF 같은 공개 등록으로 읽을 수 있다. | B 초록 "Hypotheses, thresholds and decision rules were pre-registered." → "… were pre-registered internally (timestamped, hashed design documents)." B Novelty (4) "Pre-registered hypotheses" → "Internally pre-registered hypotheses". 국문 초록도 "내부 사전등록"으로. |
| C2 | 선택 | 참고문헌이 논문 제목 대신 약칭(TD-MPC2, DINO-WM, CureWM, MiraBench, Cosmos Policy)만 적는다. 2쪽 제한 때문이면 그대로 둬도 되지만 심사자가 "불완전한 서지"로 볼 수 있다. | 공간이 남는 A(2쪽 하단 여백 있음)는 제목을 넣을 수 있다. B는 여백이 없어 그대로 둔다. |
| C3 | 선택 | 템플릿 규칙 "제목·초록에 기호·특수문자·수식 금지". B는 영문·국문 초록 모두 지켰다. A 영문 초록에 "%"가 두 번(median 95%, under 10%), 국문 초록(`abstract_ko.txt`)에 "100%p", "95%", "10%"가 있다. | 위험은 낮지만 규칙을 글자 그대로 따르려면 "median 95 percent", "under 10 percent", 국문 "100퍼센트포인트", "95퍼센트", "10퍼센트"로. |

---

## 1. A 포스터 (Planners Exploit Deterministic Error)

### 1.1 수치 대조 결과 (모두 일치)

| 원고 수치 | 재계산 값 | 출처 |
|---|---|---|
| det 최대 100 points (mw-button-press k=2) | D_det = 1.000 | e1_tdmpc2_cells.csv |
| 29 of 36 affected cells에서 det 1-step 오차 ≤ sh-iter | MW 14/19 + DMC 10/12 + DINO-WM 5/5 = 29/36 | 같은 파일 + e1_dinowm_cells.csv (`pt_err1h`, `rel_err1`) |
| mw-pick-place k=3: 0.22 vs 0.39, 37.8 vs 7.8 | 0.2234 vs 0.3876, 0.378 vs 0.078 | e1_tdmpc2_cells.csv |
| MW 19/19 red ≥ 0.5, median 95% | 19/19, median 0.947 | 〃 |
| DINO-WM 4/5, never worse | red 0.92, 1.00, 0.20, 0.69, 1.09; D_sh−D_det 모두 음수 | e1_dinowm_cells.csv |
| drift 비 ≤ 0.24; 지속 dither 70–130% | sh-iter 0.10/0.24/0.03, PushT k4 −0.01; sh-step 0.70/0.81/1.30, sh-ep 0.97/0.87/1.25 | e2_tdmpc2_summary.json `layer_b`, e1_dinowm_v52.json |
| 반복 1회 vs 12회 | it1: det 0.43/0.48/0.50, sh-iter 0.57/0.50/0.56. it12: sh-iter 0/0.11/0.05, det 0.47/0.44/0.43 | e2_tdmpc2_cells.csv |
| H2c 3/4 | TD-MPC2 3/3, PushT 불성립(ρ(N, D) 0.771) | e2 summary, E2_dinowm.md |
| H4a 5/6, three exactly at −0.8 | mw-push −0.8, pick-place −0.8, cheetah −1.0, walker +0.4(불성립); PushT −0.8, PointMaze −1.0 (Wall은 자격 없음) | e3_tdmpc2_summary.json, e3_dinowm_summary.json |
| H3b 10/19, 1/3 | 규칙 "19칸 중 10칸 이상" 충족; DINO-WM PointMaze만 | e5_*_summary.json |
| DMC rule Spearman 0.83 (12 tasks) | 0.827, 12과제 모두 포함 | e3_tdmpc2_final.json |
| S_det DMC 0.33–0.52, MW 0.54–0.73 | 0.330–0.522 / 0.539–0.733 | 〃 |
| DMC 5/12, walker 78.9 vs 56.9 | 5/12; 0.789 vs 0.569 (+0.221 [0.180, 0.263]) | e1_tdmpc2_cells.csv |
| H5 max 6.9%, weak Pareto | int8 3.8, int4 5.8, packed 6.9%; Pareto 사례 4개 모두 mw-push, E7 보고가 "약함"으로 기록 | e7_tdmpc2_summary.json, E7_tdmpc2.md §4.2 |
| 재측정 −3.2 ~ +2.7%, 정수 kernel fp32 대비 약 18% 느림 | posthoc 36조건 중앙값 −3.2~+2.7%; latency_rel_fp32 중앙값 17.5~18.2% | posthoc_tdmpc2.md, e7 summary |
| 922k 모의, 기울기 0.92–1.05 | 922,368회; CEM 1.02/1.05, MPPI 0.97, SAA 0.92–1.02 | E4_theory.md |
| 계획기 설정 512×6×3, 300×30×5, elite 64, top-30, Wall·PointMaze 10회 | DESIGN §6.0 | — |

### 1.2 고칠 항목

| # | 등급 | 위치 | 문제 | 제안 |
|---|---|---|---|---|
| A1 | **필수** | 초록, D. Setup, "Phenomenon and fix" | **"36"이 두 가지 뜻으로 쓰인다.** Setup의 "36 task–bit cells"는 TD-MPC2 전체 칸(12과제×3)이고, 초록과 본문의 "29 of 36 affected cells"는 손상 칸 합계(TD-MPC2 31 + DINO-WM 5)다. 독자는 같은 36으로 읽고 "36칸 전부가 affected인데 DINO-WM은 어디 갔나"로 혼란스러워진다. | Setup 끝에 한 구절: "…such cells are called affected (31 of the 36 TD-MPC2 cells and 5 of the 9 DINO-WM cells)." 그러면 뒤의 "29 of 36 affected cells"가 바로 읽힌다. 국문 검토본("손상 칸 36개 중 29개")도 같은 처리. |
| A2 | **필수** | 초록·본문의 "19" | **"19"도 두 가지 뜻이다.** "all 19 affected Meta-World cells"(E1 손상 MW 칸)와 "Other limitations"의 "10 of 19 TD-MPC2 cells"(E5의 3–4-bit 자격 칸: MW 12 + DMC 7)는 다른 집합이다. | "dither still helps in 10 of the 19 affected 3–4-bit TD-MPC2 cells (12 Meta-World, 7 DMC)". 표 1의 H3b 행은 "in most 3–4-bit cells"라 괜찮다. |
| A3 | **필수** | 초록 마지막에서 두 번째 문장 | **지연 수치의 범위가 과장돼 읽힌다.** "Dither removed … in 4 of 5 DINO-WM cells, at under 10% median extra latency"는 지연을 두 계열 모두에서 잰 것처럼 읽히지만, 지연은 TD-MPC2 4과제의 융합 정수 kernel에서만 쟀다. EasyChair 국문 초록(`abstract_ko.txt`)은 이미 "융합 정수 kernel에서"라고 쓰고 있어 영문과 어긋난다. | "…in 4 of 5 DINO-WM cells; in a fused integer kernel for TD-MPC2 the median extra latency of dither was under 10%." `main_ko.tex` 초록에도 "융합 정수 kernel에서"를 넣어 `abstract_ko.txt`와 맞춘다. |
| A4 | 권장 | "Mechanism" 문단 | **"with one iteration sh-iter is as harmful as det"는 데이터보다 약하게 쓴 것이다.** it1에서 sh-iter 손상이 det보다 크다(mw-push 0.57 vs 0.43, pick-place 0.50 vs 0.48, cheetah 0.56 vs 0.50). 그림 1(b)에서도 반복 1회의 파란 점이 주황 점 위에 있어 심사자가 바로 본다. | "with one iteration sh-iter is no better than det (it is slightly worse)". 국문 "반복 1회에서는 sh-iter가 det보다 낫지 않다". 이 편이 "반복하는 계획기가 필요하다"는 주장을 더 강하게 받친다. |
| A5 | 권장 | "Where the fix is weak" | **정의 없이 쓴 용어.** "That share is 0.33–0.52 on DMC…"의 "share"는 표 1에서만 "persistent share"로 나오고 정의(S_det = drift / OEI, det의 유리한 오차 중 반복 사이에 지속된 몫)가 본문에 없다. | 첫 문장을 "Dither should help less where less of det's favorable error persists across iterations (the persistent share, drift over OEI), and it does. That share is 0.33–0.52 on DMC…"로. 현재 문장 "where less favorable det error persists"는 "덜 유리한 오차"로도 읽혀 모호하다. |
| A6 | 권장 | "Mechanism" 마지막 문장 | "the damage of fixed dither shrinks as its table grows"의 "table"(dither 표, 독립 offset 수 N_eff)이 앞에서 정의되지 않았다. Algorithms의 sh-step 설명에도 "table"이 없다. | "as the number of independent offsets it draws (its table size) grows (5 of 6 tasks)". |
| A7 | 선택 | "Theory" | "Fresh unbiased noise has exactly zero expected drift"는 1차 근사에서 정확하다. E4 보고 §2.8은 비선형 2차(Jensen) 편향이 새 dither에도 남아 지속장처럼 이용될 수 있음을 적었다(파일럿 기준 det drift의 1~3%). 또 "fitted slopes 0.92–1.05"는 MPPI 지속장 마지막 반복의 0.81(약 20% 과대예측) 사례를 뺀 범위다. | 포스터 분량에서는 허용 범위. 고친다면 "has zero expected drift to first order". TMLR 원고에서는 둘 다 명시. |
| A8 | 선택 | D. Setup | Wall은 등록한 세 비트 수(4, 5, 6) 모두에서 det 하락이 실재하지 않아 손상 칸이 0개다. 포스터는 "4 of 5 DINO-WM cells"만 적어 독자가 Wall 결과를 알 수 없다. | 여백이 있으면 "(Wall showed no real det drop at its registered bit-widths)" 한 구절. |
| A9 | 선택 | `abstract_ko.txt` vs `main_ko.tex` | 두 국문 초록이 두 군데 다르다: (1) "융합 정수 kernel에서"의 유무, (2) "유리한 오차가 반복 사이에 덜 지속되는" vs "덜 지속되는", (3) 문장 분리("떨어뜨렸지만, …" vs "떨어뜨렸다. 그런데도 …"). | EasyChair에 넣는 `abstract_ko.txt`가 더 정확하니 `main_ko.tex`를 그쪽에 맞춘다(A3와 함께). |

### 1.3 괜찮은 점(바꾸지 말 것)
- 초록의 "in the four tasks where we tested the mechanism" 한정과 "overrates the plan it selects" 삭제(10-25 수정)는 PushT k=3·k=5의 양수 drift와 맞는 올바른 처리다.
- "never worse than det in any affected Meta-World or DINO-WM cell"은 손상 칸 전부에서 D_sh−D_det ≤ 0으로 확인된다(walker k=3은 DMC라 범위 밖).
- 표 1의 "fail (claims narrowed)", "weak Pareto case", "DMC fails"는 E7·E1 보고의 판정 그대로다.
- "A test registered after stage 1, on new seeds"는 v5 §7.2(E3 자료 전 고정)와 E3 seed 50000~에 맞다.

---

## 2. B 포스터 (Near-Miss Blindness)

### 2.1 수치·설정 대조 결과 (모두 일치)
- `paper/build/verify_numbers.py` 36행 전부 OK(초록·표 1·본문 수치, CI 포함).
- 설정값: 표적 3,600/칸(near 1,352~1,373), M2 4,184쌍·6과제(t0 468쌍은 별도), M3 49점×128장면×3 seed(r = 3.00만 100장면), DP 24개(3 seed×8단계×100 episode), M3 정책 6개, 판정기 정확도 0.9585/0.974/1.000, MC dropout K = 16·q* = 13/16 — E1_server·E1_m2·E4E5 보고와 일치.
- 판정: H1a(M3만), H1b(M1만; M3 2/3 조항), H2a(M3만, M2 미실행), H3(M2·M3) — FINAL_B §2 표와 일치. "fixed before analysis"(n_π 구간 [1, 1.2))는 v2 고정 22:54Z, E2 분석 시작 23:44Z로 사실이다.
- Novelty (1)의 CureWM 서술("경계 근처 severity 표본 부족", "policy ranking untested")은 `literature/NOVELTY_R2.md`의 근거와 맞다.

### 2.2 고칠 항목

| # | 등급 | 위치 | 문제 | 제안 |
|---|---|---|---|---|
| B1 | **필수** | A. Problem Setup ↔ B. Novelty ↔ C. Algorithms, Limitations | **far-miss 표본의 출처가 없고, M3의 far가 한 점이라는 한계가 빠졌다.** 본문은 far-miss를 r ≥ 3으로 정의하면서 반사실은 "cover r ∈ [0.8, 3] evenly", "target r ∈ [0.8, 3] uniformly"라고만 쓴다. 독자는 "far-miss가 어디서 나왔나"를 알 수 없다. 실제로는 M1이 [0.2, 0.8)과 [3, 6]에 각 300개 꼬리(far 298~339개/칸), M2가 과제당 성공 100 + far 100(합 584), **M3는 grid 끝 r = 3.00 한 점(w ≤ 0.13 장면 100개)**이다. M3의 far 한 점은 DESIGN v1.1("이 한계를 함께 적는다")과 FINAL_B §5가 명시한 한계인데, 표 1의 M3 "vs far 0.000"과 H1a "yes"(near/far 53.5)가 그 한 점에 기대므로 포스터에도 한 구절이 필요하다. | Algorithms: "Perturbed actions of successful trajectories cover r ∈ [0.8, 3] evenly, with tails below 0.8 and above 3 (M1: 300 targets each; M2: 100 successes and 100 far-misses per task; M3: far-miss is the grid end r = 3 only)". Limitations에 "M3's far band is a single r value" 추가. 2쪽 여백이 없으므로 "Capacity" 문장이나 "Policies" 문장을 그만큼 줄인다. 국문 검토본도 같은 처리. |
| B2 | **필수** | 초록 (영문·국문 동일) | **처방 문장의 범위가 모호하다.** "On the public model a training-free MC dropout discount helps only slightly, and boundary-targeted counterfactual data did not beat simple reweighting."은 두 번째 절도 공개 모델(M1)만의 결과로 읽힌다. H5a는 M1 1설정 + M3 2설정(0/3)이고, 유의하게 "더 나쁨"은 M3에서 나왔다. | "On the public model a training-free MC dropout discount helps only slightly, and in both latent models boundary-targeted counterfactual data did not beat simple reweighting." 국문: "공개 모델에서 재학습 없는 MC dropout 할인의 효과는 작았고, 두 latent 모델 모두에서 경계를 겨냥한 반사실 데이터는 단순 재가중보다 낫지 않았다." (`ABSTRACT_KO.md`의 긴·짧은 초록과 `main_ko.tex` 모두) |
| B3 | 권장 | 초록·Novelty (4) | 사전등록을 "내부"로 명시(공통 C1). | 위 C1. |
| B4 | 권장 | 제목·초록 | **제목·초록의 표현이 본문보다 세다.** 제목 "Near-Miss Blindness … Loses the Outcome Boundary", 초록 "loses the boundary"인데, 본문은 "blur", "boundary resolution"이고 공개 모델(M1)의 AUROC는 0.853, 상상 특징 판별기는 0.889로 정보가 상당 부분 남는다. "blindness"라는 말은 본문 어디에도 없고, M1의 near-miss 오판은 0.147이라 "blind"가 아니다. 이 말을 받치는 것은 M3(0.535)와 작은 예측기(1.000)다. | 두 가지 중 하나: (가) 제목 "…Loses Resolution at the Outcome Boundary", 초록 "…but loses resolution at the boundary"로 완화. (나) 제목은 두되 Problem Setup 끝에 한 구절로 용어를 정의: "we call the loss of this resolution near-miss blindness". 현 상태는 심사자가 "제목과 결과가 다르다"고 적기 쉽다. 저자 판단 사항. |
| B5 | 권장 | 그림 1(a) 캡션 | "thin vertical line: pre-registered cell"의 세로선(`make_figs.py` 67행, 회색 0.6pt, 범례 위쪽만)이 PDF에서 거의 보이지 않는다. | 선을 조금 진하게(예: linewidth 1.0, 색을 본문 회색보다 어둡게) 하거나 캡션의 그 구절을 뺀다. |
| B6 | 선택 | 초록 2번째 문장 | "judge real and imagined final frames with one judge"는 "판정기를 하나만 썼다"로 읽힐 수 있다(실제는 주 판정기 + 보조 판정기 3종). | "with the same judge". 국문 "같은 판정기로"는 이미 맞다. |
| B7 | 선택 | Limitations | M2 판정기의 상상 프레임 운영점 이동은 "Video world model" 문단에 있으나 Limitations에는 없다. | 여백이 나면 "judge domain shift on imagined frames (M2)" 추가. B1을 넣는 쪽이 우선이다. |
| B8 | 선택 | M3 문단 "fixed before analysis" | 설계 원문의 구간은 [1, 2)였고 [1, 1.2)는 E0 뒤(M3 E2 분석 전)에 사용자 승인으로 정했다. 현재 문장은 둘 다 보고하므로 투명하다. | 더 방어적으로 쓰려면 "fixed before the E2 analysis (the design band 1 ≤ r < 2 gives 0.66)". 지금 그대로도 무방. |

### 2.3 괜찮은 점(바꾸지 말 것)
- "five-step horizon" 한정, "far-misses also blur beyond H = 5 … and at every H under the object-pose error definition", "part of the boundary information is lost", "limited to the two latent models tested"는 10-24 독립 검토 반영이 정확하다.
- 표 1 캡션의 "each hold in one model only and fail the two-model rule"와 "H3 … (with a judge caveat)"는 FINAL_B §2·§3.1과 일치한다.
- "all of it added by imagination"은 M3 실제 프레임 판정 near 오판 0.000에 근거해 맞다.
- "M1 gives a partial result: a short (3,000-step) … rose only under the object-pose definition"은 E4E5 §2.1과 일치한다.

---

## 3. 제출 전 남은 체크리스트 (README 기준, 포스터 글과 별개)
- [ ] 이건명 교수님 공저자 등재 확인(A·B 공통)
- [ ] 연락 이메일(Gmail vs 학교 계정) 결정(B README)
- [ ] 위 필수 항목(A1~A3, B1~B2) 반영 뒤 `tectonic -X compile`로 두 PDF가 2쪽인지, B는 `verify_numbers.py`가 통과하는지 확인
- [ ] 저작권 양도 선택하지 않음, Track = General (Poster), 마감 2026-10-25

---

## 4. 반영 결과 (2026-10-25, 사용자 지시로 필수 항목만 반영)

| 항목 | 반영 파일 | 반영 내용 |
|---|---|---|
| A1 | A `main.tex`, `main_ko.tex` | Setup: "such cells are called affected (31 TD-MPC2 and 5 DINO-WM cells)"; 다음 문장 "In 29 of these 36 affected cells". 국문 "손상 칸이라 부른다(TD-MPC2 31칸, DINO-WM 5칸)", "이 손상 칸 36개 중 29개" |
| A2 | A `main.tex`, `main_ko.tex` | Other limitations: "10 of the 19 affected 3--4-bit TD-MPC2 cells". 국문 "3--4비트 손상 칸 19개 중 10개" |
| A3 | A `main.tex`, `main_ko.tex`, `abstract_ko.txt` | 초록: "; in a fused integer kernel (TD-MPC2) the median extra latency was under 10%". 국문 두 파일 모두 "TD-MPC2의 융합 정수 kernel에서 추가 지연의 중앙값은 10% 미만" |
| B1 | B `main.tex`, `main_ko.tex` | Algorithms: "cover r ∈ [0.8, 3] evenly plus tails below 0.8 and above 3 (M1: bisection per stratified target, 300 per tail; M2: binned selection, 100 successes and 100 far-misses per task; M3: a fixed r grid, far-miss only at its end r = 3)". Limitations: "M3's far band is one r value". 국문 대응 반영 |
| B2 | B `main.tex`, `main_ko.tex`, `ABSTRACT_KO.md`(긴 초록) | 초록: "and in both latent models boundary-targeted counterfactual data did not beat simple reweighting". 국문 "두 latent 모델 모두에서 …". 짧은 국문 초록에는 이 문장이 없어 그대로 |

- 2쪽을 지키기 위해 함께 한 것(뜻은 바꾸지 않음)
  - A: 참고문헌 앞 `\vspace{3pt}` → `1pt`. 본문 글은 줄이지 않았다. 2쪽 끝 여백 약 0.7줄.
  - B: 처음 반영 뒤 참고문헌 블록이 3쪽으로 넘어가서(약 4줄 초과) 다음을 했다. (1) 문장 축약 3곳: "so part of the loss is in the information itself", "(success only if at least q* of 16 passes agree; q* from a dev set)", "six policies with designed near-miss shares". (2) 참고문헌을 A 포스터와 같은 한 문단 run-in으로 바꾸고 본문의 `\cite`를 번호 [1]~[6]으로 바꿨다(순서는 원래 bibitem 순서와 같다). (3) 그림 폭 0.85 → 0.78, 그림·표 앞뒤 `\smallskip` 제거, `\maketitle` 아래 `-34pt` → `-40pt`, mdframed 안쪽 위아래 여백 4pt → 2pt. 국문 검토본에도 같은 처리를 해 두 파일의 구조를 맞췄다.
- 확인: 네 PDF(A·B × 영문·국문) 모두 2쪽. B `verify_numbers.py` 36개 모두 JSON과 일치. 바뀐 구절이 PDF 본문에 들어간 것을 텍스트 추출로 확인했다.
- 기록: A `paper/README.md` §3 "10-25 수정 3", B `paper/README.md` §4 끝 항목, B `ABSTRACT_KO.md` 머리 메모.
- 반영하지 않은 것: A4~A9, B3~B8(권장·선택). 위 1.2절·2.2절의 교체 문장을 그대로 쓸 수 있다.

### 4.1 권장 항목 반영 (같은 날, 사용자 지시)

| 항목 | 반영 내용 |
|---|---|
| C1 | A 초록 "In internally pre-registered experiments"; B 초록 "were pre-registered internally (timestamped, hashed design documents)", B Novelty (4) "Internally pre-registered hypotheses". 국문 모두 대응 |
| A4 | "with one iteration sh-iter is no better than det" |
| A5 | "where less of det's favorable error persists across iterations (the persistent share, drift over OEI)" |
| A6 | "as the number of independent offsets it draws (its table size) grows" |
| B3 | = C1 |
| B4 | 제목 "…Loses Resolution at the Outcome Boundary", 초록 "loses resolution at the boundary". 국문 제목 "결과 경계의 해상도를 잃는다", 국문 초록 "경계의 해상도는 잃는다" |
| B5 | 그림 1(a)의 사전등록 칸 세로선을 진하고 굵은 파선으로 다시 그렸고(`make_figs.py`), 캡션을 "dashed vertical line"으로 |

- 국문 초록 재점검: A `abstract_ko.txt`는 가운뎃점(·)과 %, %p를 없애고(템플릿 규칙, B와 같은 처리) 문장을 다듬어 다시 썼으며 `main_ko.tex` 초록을 같은 글로 맞췄다(A9 해소). B는 위 두 문장과 조사 하나만 고쳤다. 두 국문 초록 모두 한글, 영문, 숫자, 소수점, 쉼표, 괄호, 하이픈만 쓴다.
- 확인: 네 PDF 모두 2쪽, B `verify_numbers.py` 36개 통과. A는 2쪽을 지키려고 mdframed 안쪽 여백만 4pt → 2pt로 줄였다.
- 남은 선택 항목: A7, A8, B6, B7, B8, C2, C3.

### 4.2 문체 점검 (같은 날, 사용자 지시: 생성문처럼 읽히는 문장)

기준: 세미콜론으로 두 절을 잇는 문장, 콜론 뒤에 반전을 두는 구성("X, not Y"), 짧은 경구형 문장("and it does", "The damage follows"), 수사 의문문, 구어("happens to favor", "go both ways", "beat"), 대명사 주어가 불분명한 문장("It needs"). 수치와 주장은 바꾸지 않았고, 국문 초록은 이미 같은 뜻이라 두었다.

| 원고 | 전 | 후 |
|---|---|---|
| A 초록 | the structure of the rounding error, not its size, decides the damage | what matters is the structure of the rounding error rather than its size |
| A 초록 | drifts toward candidates this field happens to favor | drifts toward the candidates that this error favors |
| A 초록 | Changing only the rounding, at the same bit-width and without training, to dither … removes most of this drift | Replacing it with dither … removes most of this drift, at the same bit-width and without training |
| A 초록 | It needs an iterating planner | The fix requires an iterating planner |
| A Setup | Why does det hurt planning …, and does changing only the rounding fix it? | We ask why det hurts planning …, and whether changing only the rounding repairs it. |
| A Novelty | The principles, …, are classical; their application … are new | These principles (…) are classical; their application …, the two-axis separation and the effect size are new |
| A Algorithms | the prescription / are official / exactly zero | the proposed fix / are the official defaults / zero |
| A Experiments | The damage follows: / Where the fix is weak. … and it does. / hurts on walker / finds this share predicts / low bits gave no speedup | Damage tracks the drift: / Limits of the fix. If dither works by removing persistent exploitation, it should help less … / is harmful on walker / confirms that this share predicts / the low-bit kernels gave no speed-up |
| B 초록 | …; we ask which failures are misjudged / with one judge / go both ways rather than toward optimism / all of it added / did not beat / so our claims are limited to | … . This work asks which failures are misjudged / with the same judge / are two-sided rather than optimistic / all of which is added / did not outperform / so we limit our claims to |
| B Setup·Novelty·Algorithms | …; we ask how this depends on how far a failure lands from the boundary / 1,300+ / used only with held-out accuracy at least 0.95 | . We ask how this error depends on the distance of a failure from the boundary / over 1,300 / used only if its held-out accuracy is at least 0.95 |
| B Experiments | The errors go both ways / turns blur into optimism / not lost information / yet their order is kept / never beat reweighting (0 of 3 settings; …) / because M1's bias is small / one r value | The errors are two-sided / converts the blur into optimism / rather than a loss of information / but their ordering is preserved / did not outperform reweighting in any of the 3 settings (…) / since M1's bias is itself small / a single r value |

- 2쪽 유지: A는 그림 폭 0.90, 제목 아래 간격 -36pt로 맞췄다. B는 여유 안에서 들어갔다. 네 PDF 모두 2쪽, B `verify_numbers.py` 통과.
- 그대로 둔 것: 표 1의 전보문체 항목(표는 원래 그렇게 쓴다), B Limitations의 세미콜론 나열(포스터 분량 때문), 국문 검토본(`main_ko.tex`)의 짧은 문장들(제출본이 아님).

---

## 5. 판단: 이 실험을 지금의 구조로 쓰는 것이 최선인가

템플릿이 절 제목(A. Problem Setup, B. Novelty, C. Algorithms, D. Experiments)을 고정하므로, 판단할 것은 세 가지다. (가) 무엇을 앞세우는가, (나) 그림과 표가 어떤 증거를 지는가, (다) 제목·초록의 틀이 본문의 증거와 맞는가.

### 5.1 A — 지금 구조가 맞다. 바꾸지 않는 것을 권한다.

- **앞세운 것이 가장 튼튼한 증거다.** 초록은 현상(1-step 오차로 설명되지 않는 손상) → 명제(크기가 아니라 구조) → 기전(지속 오차장과 이동) → 처방(sh-iter) → 경계(반복형 계획기, DMC) 순서다. 이 중 "크기가 아니라 구조"는 기전 과제 4개가 아니라 E1·E3 전체 칸에서 나온다. 항목별 오차 분포가 같은 변형(ind, sh-iter, sh-step, sh-ep)끼리 손상이 다르고(H3a 19/19 대 지속 dither의 drift 70~130%), 1-step 오차가 더 작은 det가 더 크게 망가진다(H1, 29/36). 기전 과제 4개로 한정한 것은 drift의 정량(H2a′)뿐이고, 초록도 그렇게 한정해 썼다. 따라서 기전을 앞세운 틀이 증거보다 앞서 나가지 않는다.
- **그림·표의 분담이 맞다.** 그림 1(a)는 처방(모든 손상 칸), 1(b)는 기전(반복 수 곡선), 표 1은 사전등록 장부다. 실패한 가설(H2a·H2b·H4b, DMC, DINO-WM H3b)이 장부에 그대로 있어 심사자가 "좋은 것만 골랐다"고 보기 어렵다. 국내 학회 포스터에서는 드문 형식이지만 이 연구의 강점이라 유지한다.
- **"처방이 약한 곳"을 표 앞에 둔 것도 맞다.** DMC를 한계가 아니라 기전이 예측하는 경계(S_det 0.33~0.52, 사전등록한 DMC 규칙 0.83)로 쓴 10-24 결정이 구조를 살린다.
- **대안을 검토한 결과.** (i) 기전(E2)을 실험 절의 머리에 두고 처방을 따름정리로 쓰는 구조는 더 나쁘다. 기전 증거가 처방 증거보다 좁다. (ii) 현상과 처방만 쓰고 기전을 빼는 구조는 "왜 dither인가"를 잃고, DMC 결과를 설명할 수 없게 된다. 지금 구조가 둘 사이의 올바른 자리다.
- **TMLR 때 바꿀 것(포스터에서는 아님).** 진단(H1, 계획 편차)·기전(drift, 영역 효과 drift_ref)·처방(H3a·H3b·H5)을 절로 분리하고, H2a·H2b의 불성립을 "같은 draw의 OEI에는 winner's curse가 섞인다"는 E4의 방법론 결과로 승격해 쓴다. 포스터에서는 표 1 한 줄로 충분하다.

### 5.2 B — 절 구조는 유지하되, 틀을 "near-miss blindness"에서 "흐림 → 낙관의 두 단계"로 옮기는 것이 증거와 더 맞다.

- **지금 틀의 문제.** 제목·초록은 원래 가설의 언어(near-miss에 특이한 낙관 = blindness, 경계를 "잃는다")로 쓰여 있는데, 그 가설(H1a·H1b·H2a의 2-모델 규칙)은 성립하지 않았다. 살아남은 증거는 두 개다. (1) M1: latent 상상은 경계 해상도를 양방향으로 잃는다(AUROC 0.953 → 0.853, 상상 특징에도 0.889만 남음). (2) M3: 성공 편중 학습이 그 흐림을 한쪽(낙관)으로 민다(용량-반응, 모두 상상에서 생김, 정책 과대 추정이 near-miss 비중을 따름). 본문의 단락 제목("Success-skewed data turns blur into optimism")은 이미 이 두 단계 틀로 쓰였는데, 제목과 초록 첫머리만 옛 틀에 남아 있다.
- **두 단계 틀로 옮기면 얻는 것.** "M1과 M3가 주장의 서로 다른 부분을 받친다"는 한계가 설계로 읽힌다. 1단계(해상도 손실)는 공개 모델에서, 2단계(데이터 균형의 인과)는 통제 모델에서 각각 분리해 보인 것이기 때문이다. 다만 "한 모델에서 두 단계를 모두 보이지는 못했다"는 한계는 그대로 적어야 한다. H3(상상 몫, M2·M3)은 2단계의 보조 증거로 자연스럽게 붙고, M2의 no-go는 "경계 특이성은 비디오 모델로 일반화되지 않았다"로 자리가 생긴다.
- **오늘 할 수 있는 최소 버전**은 2.2절 B4다. 제목을 "…Loses Resolution at the Outcome Boundary"로 완화하고 초록의 "loses the boundary"를 "loses resolution at the boundary"로 바꾸거나, Problem Setup 끝에 용어 정의 한 구절을 넣는다. 절의 순서를 바꾸지 않아도 된다.
- **실험 절의 순서(TMLR 때).** 지금은 표 1(판정 장부)이 D. Experiments의 첫 요소라 독자가 결과를 보기 전에 "불성립" 여섯 개를 먼저 읽는다. A처럼 서술(M1 → M3 → 처방 → 비디오) 뒤에 장부를 두는 쪽이 같은 투명성으로 더 잘 읽힌다. 오늘은 2쪽 배치 위험이 커서 권하지 않는다.
- **작은 것.** Problem Setup에 있는 지표 정의(상상 몫 식, o_π, n_π)는 Algorithms로 옮기면 Problem Setup이 문제 서술로 읽힌다. 그림 1의 분담((a) M1 해상도 대 상상 길이, (b) M3 p별 곡선)은 두 단계 틀과 정확히 맞으니 그대로 둔다.

### 5.3 공통

- A와 B를 따로 내는 것이 맞다. 질문이 다르고(계획기의 오차 이용 / 평가의 경계 해상도), 뒤의 투고처(TMLR·ICLR WS)도 따로 간다.
- 2쪽 포스터는 두 연구 모두에서 "사전등록한 주장과 부정 결과의 공개 기록" 역할을 한다. 논문집에 실리지 않고 저작권을 넘기지 않으므로 TMLR과 충돌하지 않는다(각 README §2·§5).
