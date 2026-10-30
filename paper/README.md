# B 논문 원고 — Near-Miss Blindness

| 폴더 | 대상 | 상태 (2026-10-24) |
|---|---|---|
| [`jkaia2026/`](jkaia2026/) | 2026 한국인공지능학회 추계학술대회(JKAIA 2026), **General (Poster) 트랙** | 초안 v1. 영문 제출본 `main.tex`·`main.pdf`(2쪽), 한국어 대조본 `main_ko.tex`·`main_ko.pdf`, EasyChair 입력값과 국문 초록 [`ABSTRACT_KO.md`](jkaia2026/ABSTRACT_KO.md) |
| [`explainer/`](explainer/) | 연구설명 문서(DriveWAM 연구설명과 같은 형식, 한국어 25쪽) | 2026-10-26 완성. docx·pdf는 `research/papers/Near-Miss Blindness/`에도 있다 |
| `tmlr/` (예정) | TMLR | 미착수. 11월 중순 제출 목표 |
| `iclr2027_ws/` (예정) | ICLR 2027 워크숍 | 미착수. 워크숍 목록은 2026-11-29 공개 예정 |

- 공통 파일
  - [`figures/fig_boundary.pdf`](figures/fig_boundary.pdf): (a) M1 경계 해상도 대 상상 길이, (b) M3 학습 실패율별 P(성공 판정 | r)
  - [`build/make_figs.py`](build/make_figs.py): 결과 JSON만 읽어 그림을 다시 그린다. 새 계산은 없다.
  - [`build/verify_numbers.py`](build/verify_numbers.py): 원고에 쓴 수치 36개를 결과 JSON에서 다시 계산해 `main.tex`·`main_ko.tex`와 대조한다. 원고를 고친 뒤 반드시 돌린다.
- 빌드(맥북에서 몇 초)

```bash
cd "/Users/song-yonghwi/Documents/physical ai - Neural Networks/B_near_miss_blindness/paper/jkaia2026" && tectonic -X compile main.tex && tectonic -X compile main_ko.tex
```

```bash
cd "/Users/song-yonghwi/Documents/physical ai - Neural Networks/B_near_miss_blindness" && python3 paper/build/make_figs.py && python3 paper/build/verify_numbers.py
```

---

## 1. JKAIA 2026 규정 (2026-10-24 학회 홈페이지에서 직접 확인)

- 출처
  - [논문제출 안내](https://aiassociation.kr/Conference/ConferenceView.asp?AC=0&CODE=CC20260901&CpPage=377)
  - [논문 모집](https://aiassociation.kr/Conference/ConferenceView.asp?AC=0&CODE=CC20260901&CpPage=376)의 CFP 포스터 이미지
  - 학회소식 게시판의 [2026 추계학술대회 CFP 공지](https://aiassociation.kr/board/board.asp?b_code=290&Action=content&GotoPage=1&B_CATE=BBS1)에 붙은 투고 절차 안내 이미지
  - [사전등록](https://aiassociation.kr/Conference/ConferenceRegistration.asp?AC=0&CODE=CC20260901&CpPage=R) 페이지
  - 포스터 트랙 템플릿 zip(아래)
- DriveWAM 폴더의 기존 조사(`DriveWAM_Lightweight_Action_Readout/docs/paper/README.md` §1, 2026-10-15)와 비교했다. 바뀐 규정은 없었다. 이번에 더 확인한 것은 템플릿 원문의 초록 규칙, 채택 시점의 저작권 조항, 등록비, 연장 공지 여부다.

| 항목 | 내용 |
|---|---|
| 일시·장소 | 2026-11-19(목)~20(금), 서울대학교 호암교수회관 베리타스홀 |
| **논문 제출 마감** | **2026-10-25(일)**. 마감 시각과 시간대는 안내에 없다. 10-24 23시(KST) 기준으로 학회소식 게시판에 연장 공지가 없다(2024년에는 10-21에 연장 공지가 있었다) |
| 심사결과 통보 | 2026-11-09(월) |
| 제출 | EasyChair <https://easychair.org/conferences/?conf=jkaia2026> (계정 필요). 트랙은 General (Full paper)와 General (Poster) 중에서 고른다 |
| 언어 | 논문은 영문으로 쓴다. EasyChair에는 **제목을 영문, 초록을 국문**으로 넣는다 |
| 심사 | single-blind. 저자를 명기한다 |
| 포스터 트랙 양식 | LaTeX 템플릿을 쓴다. 초록은 선택이다. 템플릿을 1쪽이나 2쪽으로 늘려 쓴다. 제목·저자·소속·초록(선택)을 포함한 2장 분량 요약본 PDF를 낸다 |
| 템플릿 원문의 규칙 | 초록은 10~20줄이다. **제목과 초록에는 기호, 특수문자, 각주, 수식을 쓰지 않는다.** 구성은 Abstract, Keywords, A. Problem Setup, B. Novelty, C. Algorithms, D. Experiments를 하나의 틀 안에 넣는 것이다 |
| 저작권 | 양도는 선택이다(CFP: "국제학회 투고계획을 고려"). 투고 절차 안내 이미지 4번: 포스터·구두 발표 뒤 **논문집에 포함되지 않고 국제 학회 제출·게재가 가능**하다. 논문제출 안내: 채택 시점에 다른 학회·저널로 저작권이 양도되어 있으면 안 된다(채택 통보나 발표 준비 같은 가확정 포함). 저작권 교신은 채택 뒤 학회 이메일로 한다 |
| 시상·JARA | 상금(총 1,000만 원)과 JARA 출판 심사는 General (Full Paper) 트랙만 해당한다 |
| 등록 | 사전등록은 11-09(월)까지. 학생 회원 17만 원, 학생 비회원 22만 원(현장 22만·27만 원). 학생 연회비는 3만 원이라 회원 가입 뒤 등록하는 쪽이 2만 원 싸다. 환불은 11-12(목) 24:00까지 이메일로 요청한다 |

- **템플릿**: 포스터 트랙 링크는 `https://aiassociation.kr/UploadData/Editor/Conference/202610/4C732AD210FF48878CC64322E9E5470A.zip`이다. Full Paper 트랙 링크는 다른 파일(`202310/706A99AD...zip`)이다.
  - 2026-10-24에 포스터 zip을 다시 받아 확인했다(scratchpad에만 두고 프로젝트에는 `kaia.cls`만 복사했다).
  - `kaia.cls` sha256 `76b64e11061d554cecf7f50f786678ce999729f147cd9407033177348cf3cd21`. DriveWAM 사본과 바이트 단위로 같다.
- **템플릿 대비 바꾼 점**(DriveWAM 포스터와 같은 처리, 파일 머리 주석에도 적었다)
  - 템플릿의 `\fbox{\parbox}`는 쪽을 넘기지 못한다. 그래서 2쪽 요약을 위해 `mdframed`로 같은 모양의 테두리를 그렸다.
  - `kaia.cls`의 `\maketitle`이 `\twocolumn[...]`을 부른다. 그래서 `\twocolumn`을 1단으로 재정의해 템플릿 PDF와 같은 1단 모양을 지켰다.
  - tectonic(XeTeX)에서는 Times(`mathptmx`)가 매핑되지 않는다. 그래서 `\ifXeTeX`일 때만 시스템 Times New Roman을 쓴다. pdflatex·Overleaf에서는 클래스 기본값이 그대로 적용된다.
- **확인하지 못한 것**
  - EasyChair 제출 화면은 로그인해야 보여서 초록 글자 수 제한과 필수 항목을 보지 못했다. 그래서 `ABSTRACT_KO.md`에 약 390자 짧은 초록도 준비했다.
  - 마감 시각과 시간대가 안내에 없다. 10-25(일) 낮에 제출하기를 권한다.
  - 한 사람이 포스터를 두 편(DriveWAM과 B) 낼 때 논문마다 등록이 필요한지 안내에 없다. kaia@aiassociation.kr에 묻는다.

## 2. TMLR·ICLR 2027 워크숍과 함께 내기 위한 조건

- **반드시 General (Poster) 트랙으로 내고 저작권 양도를 선택하지 않는다.** 그래야 논문집에 실리지 않는 발표가 된다.
- TMLR(DriveWAM `docs/paper/README.md` §2, 2026-10-15 확인)
  - 다른 archival venue에 게재·채택되었거나 동시에 제출된 내용은 재사용할 수 없다.
  - 명시적인 비archival venue(워크숍 등)와 arXiv 공개는 허용된다.
  - 그래서 논문집에 실리지 않는 KAIA 포스터는 허용 범위다. TMLR 제출 때 이 사실을 밝힐 수 있게 위 1절의 출처(투고 절차 안내 이미지 4번)를 보관한다.
- KAIA의 "채택 시점 저작권" 조항: KAIA 통보는 11-09이고 TMLR은 그 뒤에 낸다. TMLR은 저작권을 양도받지 않는다(CC BY 4.0). 그래서 충돌하지 않는다.
- TMLR은 이중맹검이다. TMLR 원고에 KAIA 포스터를 실명으로 링크하지 않는다.
- ICLR 2027 워크숍: 목록 공개는 11-29, 원고 마감은 2027-02-01 무렵이다(메모리의 10-24 조사). 워크숍마다 "다른 곳에서 심사 중인 논문"에 대한 규정이 다르다. TMLR 심사 중에 낼 수 있는지 그때 확인한다.
- **TMLR 전에 읽을 관련 연구**(2026-10-24 arXiv 검색, 제목·초록만 봄). 실패 거리별 곡선이나 정책 순위를 재는 직접 경쟁 연구는 찾지 못했다. 아래는 "성공 편향 환각과 실패 데이터 처방" 계열로 관련 연구 절에 더한다.
  - ProWorld (arXiv:2610.15294, 10-10): 정책 평가용 multi-view 월드모델. 놓친 grasp를 성공한 들어 올리기로 그리는 환각을 실패 상호작용 데이터로 줄인다.
  - WoVR+ (arXiv:2610.17922, 10-13), HaWMPO (arXiv:2609.09941, 09-09): 월드모델 안 RL의 환각 문제

## 2.1 투고 전에 다시 확인할 것 (TMLR용)

- FINAL_B §9의 집필 할 일(H3 그림에 M2 칸 추가, 사전등록 부록, 재현 부록, 수치 정의 통일 D4·D5)을 따른다.

## 3. 제출 전에 결정·확인할 것

- [x] **공저자**: 이건명 교수님(Keon Myung Lee, 충북대학교 소프트웨어학부, kmlee@cbnu.ac.kr)으로 정했다(2026-10-24 사용자 지시). 영문 표기·소속·이메일은 JEV 논문(KIIS 2026 추계), `CITATION.cff`, CV, A 포스터 초안과 같다. 소속 줄은 2쪽에 맞추려고 A 포스터처럼 도시명 없이 "Chungbuk National University, Korea"로 썼다.
- [ ] 이건명 교수님께 공저자 등재를 확인받는다.
- [ ] 연락 이메일: 개인 Gmail 그대로 둘지, 학교 계정으로 바꿀지 정한다.
- [ ] `main_ko.pdf`를 읽고 고칠 곳을 정한다. 고친 문단은 `main.tex`의 같은 자리에 반영한다. 반영 뒤 `verify_numbers.py`를 돌리고 쪽수가 2쪽인지 확인한다. 현재 둘째 쪽이 거의 꽉 찼다(여백 1줄 미만). 문장을 늘리면 다른 곳을 그만큼 줄여야 한다.
- [ ] EasyChair 입력: Track General (Poster), 영문 제목, 국문 초록, 키워드, `main.pdf` 업로드. 값은 [`jkaia2026/ABSTRACT_KO.md`](jkaia2026/ABSTRACT_KO.md)에 있다.
- [ ] 저작권 양도는 선택하지 않는다.
- [ ] 10-25(일) 안에 제출한다.
- [ ] 채택되면 사전등록(11-09까지), 포스터 규격 안내, 등록비 중복 여부를 확인한다.

## 4. 원고의 주장 범위

[`../main/DESIGN_main.md`](../main/DESIGN_main.md) §8·§10·§11과 [`../main/reports/FINAL_B.md`](../main/reports/FINAL_B.md) §3을 따랐다.

- 2026-10-24 독립 검토(별도 에이전트, FINAL_B·DESIGN_main·단계 보고와 대조)에서 나온 필수 4건을 모두 반영했다.
  - near-miss 특이성의 범위: 초록에 "five-step horizon", 본문에 H > 5와 물체 자세 정의(r_obj)에서 far도 흐려진다는 단서(0.058~0.129)
  - 기울기 0.031은 near-miss 하락폭의 기울기임을 명시
  - M2 H3 단서: 상상 몫의 일부가 판정기 영역 차이일 수 있음(본문과 표 캡션)
  - 1차(H1a·H1b·H2a)와 탐색(나머지) 가설 구분(Novelty (4), 표 캡션, 단락 제목)
  - 권장 사항도 반영했다: M3 H2a의 n 구간(1 ≤ r < 1.2, 설계 구간 1 ≤ r < 2에서는 0.66), M1 미세조정의 한계(3,000 step, r_obj에서만 증가), "정보의 일부", DINO-WM 역할 표현, "has been reported", CureWM 처방 인용, 영문 표현 몇 곳
- 원고에 쓴 주장
  - latent 월드모델의 상상은 결과 경계의 해상도를 잃는다(M1: H1b·H4b 성립). near-miss 특이성은 원 정의 r의 H ≤ 5에서만 쓴다.
  - M1의 흐림은 낙관이 아니라 양방향이다(성공 판정 0.862 → 0.710).
  - 성공 편중 학습 데이터가 흐림을 낙관으로 바꾼다(M3 인과 결과: H4a·H1a·H3·H2a 성립). M1의 미세조정은 부분 결과로 적었다.
  - 재학습 없는 MC dropout 할인은 작은 보정이다(M1, H5b).
  - near-miss 오판의 주된 원천은 상상이다(탐색 가설 H3, M2·M3). 이 문장에는 M2 판정기 단서를 붙였다.
  - 예측기 용량 결과는 "not pre-registered"로 표시했다.
  - 2쪽에 맞추려고 뺀 것: World-VLA-Loop 인용, Statistics 항목(95% CI 표기는 첫 구간에 남김), Holm 보정 언급. TMLR 원고에서는 되살린다.
- 원고에 함께 적은 부정 결과: M2 no-go(경계 특이성 불성립), 1차 가설군이 "모델 2개 이상" 기준에 못 미침(표 1 캡션), H5a 부정 결과(경계 겨냥 반사실은 처방으로 제안하지 않음), 순위 역전 없음. checkpoint 선택 미시험(E6 생략)은 포스터에서 뺐다(주장하지 않는 내용). TMLR 원고에는 쓴다.
- 쓰지 않은 주장(FINAL_B §3.3): "월드모델은 낙관적이다"라는 일반 명제, 비디오 월드모델에서의 near-miss blindness, 순위 역전, 경계 겨냥 데이터 처방, checkpoint 선택 regret, "최초 발견"

- 2026-10-25 2차 독립 검토(`../workspace/JKAIA_POSTER_REVIEW_2026-10-25.md`)의 필수 항목 B1·B2를 반영했다(사용자 지시).
  - B1: Algorithms의 반사실 문장에 far-miss 표본의 출처를 넣었다: "cover r ∈ [0.8, 3] evenly plus tails below 0.8 and above 3 (M1: …, 300 per tail; M2: …, 100 successes and 100 far-misses per task; M3: a fixed r grid, far-miss only at its end r = 3)". Limitations에 "M3's far band is one r value"를 더했다. 근거: DESIGN_main §0.2(꼬리 각 300개), §0.4(과제당 성공 100·far 100), v1.1(M3 far는 r = 3.00 한 점), FINAL_B §5.
  - B2: 초록의 처방 문장을 "…and in both latent models boundary-targeted counterfactual data did not beat simple reweighting"으로 바꿨다(H5a는 M1 1설정 + M3 2설정). `main_ko.tex`와 `ABSTRACT_KO.md`의 긴 초록도 "두 latent 모델 모두에서"로 바꿨다(짧은 초록에는 이 문장이 없다).
  - 2쪽을 지키려고 글은 뜻이 같은 범위에서만 줄였다: "so part of the loss is in the information itself", "(success only if at least q* of 16 passes agree; q* from a dev set)", "six policies with designed near-miss shares". 배치는 참고문헌을 A 포스터와 같은 한 문단 run-in(`\cite` 대신 본문에 번호)으로 바꾸고, 그림 폭 0.85 → 0.72, 그림·표 앞뒤 `\smallskip` 제거, `\maketitle` 뒤 `-34pt` → `-40pt`, mdframed 안쪽 여백 2pt, 표 `\arraystretch` 0.92로 했다. `verify_numbers.py`는 36개 모두 통과, 두 PDF 모두 2쪽이다.
  - 권장·선택 항목(B3~B8: 내부 사전등록 표기, 제목 완화, 그림 세로선, "the same judge")은 반영하지 않았다. 검토 보고서에 교체 문장이 있다.

- 2026-10-25 권장 항목 C1·B3~B5 반영(사용자 지시)과 국문 초록 재점검
  - C1/B3: 초록 "were pre-registered internally (timestamped, hashed design documents)", Novelty (4) "Internally pre-registered hypotheses". 국문 "시각과 해시를 기록한 내부 설계 문서로 사전등록했다", "내부 사전등록".
  - B4: 제목을 "Near-Miss Blindness: Latent World-Model Imagination Loses Resolution at the Outcome Boundary"로, 초록을 "loses resolution at the boundary"로 완화했다(pdftitle, `main_ko.tex`의 국문·영문 제목, `ABSTRACT_KO.md`의 Title 행 포함). 본문의 AUROC 0.853과 상상 특징 판별기 0.889에 맞춘 표현이다. 용어 정의를 본문에 넣는 대안(나)은 하지 않았다.
  - B5: `build/make_figs.py`의 사전등록 칸 세로선을 진하고 굵게(#555555, 1.0pt, 파선) 바꾸고 그림을 다시 만들었다(결과 JSON만 읽음). 캡션은 "dashed vertical line: pre-registered cell"(국문 "세로 파선").
  - 국문 초록: 위 두 문장 외에 "성공과 먼 실패는 여전히 구분하지만"의 조사를 "실패를"로 고쳤다. 짧은 초록은 "내부 설계 문서로 사전등록한 실험으로 조사했다"로 맞췄다. 글자 수는 `ABSTRACT_KO.md` 제목에 다시 적었다.
  - 두 PDF 모두 2쪽, `verify_numbers.py` 36개 통과. 반영하지 않은 것: B6("the same judge"), B7(Limitations의 판정기 영역 차이), B8, C2, C3.

- 2026-10-25 문장 정리(사용자 지시: 생성문처럼 읽히는 문장. 수치·주장 불변)
  - 초록: "… failed executions. This work asks which failures are misjudged.", "with the same judge"(B6 해소), "the errors are two-sided rather than optimistic", "all of which is added by imagination", "did not outperform simple reweighting", "so we limit our claims to".
  - Problem Setup: "We ask how this error depends on the distance of a failure from the boundary." Novelty: "(over 1,300 near-misses per cell)". Algorithms: "used only if its held-out accuracy is at least 0.95".
  - Experiments: "The errors are two-sided:", 단락 제목 "Success-skewed data converts the blur into optimism", "rather than a loss of information", "but their ordering is preserved", "did not outperform reweighting in any of the 3 settings", "a small gain, since M1's bias is itself small", "a single r value".
  - 국문 초록은 바꾸지 않았다(이미 같은 뜻). 2쪽 유지, `verify_numbers.py` 통과.

- 2026-10-25 배치 복원과 작은 수정(사용자 지시)
  - 배치: 2쪽 끝에 약 3줄이 비어 있어서 앞선 축소의 일부를 되돌렸다. 그림 폭 0.78 → 0.85, `\maketitle` 아래 −40pt → −34pt, 틀 위쪽 안쪽 여백 2pt → 3pt, 그림 앞뒤와 표 뒤 `\smallskip`(2쪽 끝 여유 약 2줄). 틀 위아래 여백을 4pt까지 되돌리면 3쪽이 되어 그 이상은 하지 않았다. 참고문헌 한 문단 배치는 그대로 뒀다.
  - 문장: M2 "up to 100 far-misses per task"(t2는 후보가 84개뿐, `results/E1_m2/analysis/e1_analysis.json` composition), 초록 "This work asks" → "We ask"(다음 문장들의 주어 We와 맞춤). 국문 대조본은 "far-miss 최대 100개"로 맞췄다. 국문 "본 연구는 … 묻는다"는 그대로 뒀다.
  - `ABSTRACT_KO.md` 용어 대응표의 "the errors go both ways"를 현재 영문 "the errors are two-sided"로 바꿨다.
  - 확인: 두 PDF 모두 2쪽, `verify_numbers.py` 36개 통과. 1쪽은 C절 첫 줄까지 꽉 차 있다. 초록이나 A·B절이 한 줄 늘면 C절 제목이 2쪽으로 밀려 3쪽이 된다.

- 2026-10-25 Limitations 줄 삭제(사용자 지시: 끝에 붙인 점검표 같은 문장)
  - 영문 "Limitations. …"와 국문 "한계. …" 줄을 지웠다. 항목은 본문의 근거 옆에 이미 있다.
    - M3 far 한 점: Algorithms "M3: a fixed r grid, far-miss only at its end r = 3"
    - 순위 역전 없음: M3 문단 "but their ordering is preserved"
    - M1·M3가 다른 부분을 받침: M3 문단 "M1 gives a partial result", 초록 "limit our claims to the two latent models tested"
    - 시뮬레이션만: Algorithms "the simulator gives the true r"
  - checkpoint 선택 미시험(E6)만 포스터에서 빠진다. 포스터가 그에 대해 아무것도 주장하지 않으므로 숨기는 것은 아니다. TMLR 원고의 한계 절에 쓴다.
  - 확인: 두 PDF 모두 2쪽, `verify_numbers.py` 36개 통과.

## 5. 원고 각 부분의 근거

| 원고 부분 | 근거 문서 | 수치 출처(JSON) |
|---|---|---|
| A. Problem Setup, 모델 M1~M3 | DESIGN_main §0·§2, FINAL_B §1.1 | — |
| B. Novelty, 선행연구 | EXPERIMENT_DESIGN §2, `literature/NOVELTY_R2.md`; arXiv 초록 페이지에서 저자·번호 재확인(2026-10-24: 2411.04983, 2601.16163, 2610.09134, 2605.29360, 2602.06508) | — |
| C. Algorithms | DESIGN_main §0·§3·§9, E1_server·E1_m2·E4E5 보고 | `E1_server/m1/judges_e1.json`, `E1_m2/judge_a/cnn_held_e1.json` |
| 그림 1(a) | E1_server §3, E4E5 §2.3 | `E1_server/m1/e1_m1_analysis.json`, `E4_server/e4_analysis.json` |
| 그림 1(b) | E1_server §4 | `E1_server/m3/e1_m3_analysis.json` |
| 표 1 | FINAL_B §2 | e1_m1·e1_m3·`E1_m2/analysis/e1_analysis.json`·`h3_m2.json`·e2_m1·e2_m3·e3_m1 |
| M3 용량-반응, 용량 sweep | E4E5 §2.2·§2.4 | `E4_server/e4_analysis.json` |
| 처방 | E4E5 §3 | `E5_server/e5_analysis.json` |
