# B 논문 실험설계안 — Near-Miss Blindness: 월드모델의 상상은 결과 경계를 잃는다

- 작성: 2026-10-21 · 상태: 본 실험 설계 v1(사전등록 초안)
- **2026-10-22 진행 상황**: 본 실험 E1에서 M2(Cosmos Policy)가 사전 규칙상 no-go로 나왔다. 사용자 승인으로 주장을 latent 월드모델(M1·M3)로 좁혔고, E4·E5까지만 진행한다. 상세: [main/DESIGN_main.md](main/DESIGN_main.md) §8
- **2026-10-22 실험 단계 완료(E0~E5)**: 사용자가 최종 서술 범위를 승인했다. 상세: [main/DESIGN_main.md](main/DESIGN_main.md) §10
  - H4a는 통제 모델 M3의 인과 결과로, M1은 부분 결과로 쓴다.
  - H5a(경계 겨냥 반사실 자료)는 처방 주장에서 빼고 부정 결과로 보고한다.
  - H5b는 작은 평가 쪽 보정으로 쓴다.
  - 예측기 용량 결과는 사전등록 밖 서술로 넣는다.
  - H3(상상이 오판의 원천)은 사전등록 규칙대로 M2·M3 두 모델에서 성립했다(M2 상상 몫 0.829, 판정기 영역 차이 단서 포함; [main/DESIGN_main.md](main/DESIGN_main.md) §11).
- 근거
  - 파일럿 5개([pilots/](pilots/), 합계 약 $2.3)
  - 노벨티 재조사 [../literature/NOVELTY_R2.md](../literature/NOVELTY_R2.md) §2
  - 2026-10-21 추가 검색(2.1절 표의 HF-WM, FARM, CAG)
- 원칙: 본 실행 전에 이 문서의 가설·지표·판정 규칙을 `DESIGN_main.md`로 고정한다. 그 뒤에 바꾼 것은 사후 분석으로 표시한다.
- **시간 위험**: 같은 현상을 다룬 CureWM(2026-10-06)이 정책 순위를 미검증 과제로 남겼다. 빨리 움직여야 한다. 목표는 12월 중순 arXiv다.

---

## 0. 한눈에 보기

**문제**
- 월드모델로 로봇 정책을 평가하고 고르는 일이 늘고 있다. 예: Ctrl-World, WorldGym, Veo 시뮬레이터, DreamDojo.
- 월드모델이 실패를 성공으로 그린다는 보고는 이미 많다. 예: MiraBench, CureWM, World-VLA-Loop, dWorldEval.
- 하지만 **어떤 실패를** 성공으로 그리는지, 그것이 **정책 평가와 checkpoint 선택을 어떻게 왜곡하는지**는 재지 않았다.

**주장(한 문장)**
월드모델의 상상은 성공 경계 바로 근처(near-miss)의 정보를 선택적으로 잃는다. 그래서 상상이 실패를 성공으로 판정할 확률은 실패가 경계에서 얼마나 먼지의 함수다. 정책 평가의 오차는 정책의 성공률이 아니라 실패의 기하(near-miss 비중)를 따른다. 실제 성공률이 같은 정책도 상상에서는 순위가 갈리고, 상관계수와 MMRV 같은 표준 평가 지표는 이것을 거의 벌하지 못한다.

**기여(예정)**
1. **실패 거리 곡선**: 경계 근처까지 촘촘히 잰 P(상상 성공 | 실제 실패, 실패 거리 r). 공개 월드모델 2종 이상과 통제 모델 1종에서 잰다.
2. **정책 수준 법칙**: 학습된 정책 checkpoint에서 과대 추정이 near-miss 비중을 따른다. 실제 성공률이 비슷한 정책 사이에서는 순위가 역전된다. 표준 지표의 맹점도 보인다.
3. **원천 분해**: 판정기(VLM, value head)의 지각 흐림과 상상(예측)의 정보 손실을 분리한다.
4. **기전과 처방**
   - 학습 데이터의 성공 편중이 "양방향 흐림"을 "낙관"으로 바꾼다는 인과 시험
   - 노출량을 맞춘 데이터 처방
   - 재학습 없는 평가 쪽 처방(경계 인지 할인)
   - 경계 해상도 지표

**지금까지의 증거(요약, 자세히는 5장)**

| 설정 | 핵심 수치 |
|---|---|
| 직접 만든 2D 집어 들기, 상태 · DINOv2 예측기, 학습 데이터 실패율 1% (R2-4, R2-6) | near-miss 실패를 성공으로 92.5~99.6% 예측, far-miss는 5~6% |
| DINO-WM식 latent 비디오 예측기로 정책 평가, 실패율 3% (R2-9) | 실제 성공률 52.6%와 48.2%인 두 정책이 상상에서 98.5%와 58.0%. 정책 6개에서 near-miss 비중과 과대 추정의 상관 r = 0.95 |
| **공개 DINO-WM PushT** (R2-12) | P(상상 성공 \| near-miss) 0.383 대 far-miss 0.002(목표 거리 판정기). 성공 vs near-miss 판별 AUROC가 실제 0.967 → 상상 5단계 0.880 → 10단계 0.676. 성공 vs far-miss는 0.996(10단계 0.944) |

**남은 질문**
- 정책 평가에 실제로 쓰이는 비디오 월드모델(Cosmos Policy LIBERO)에서도 성립하는가
- 학습된 정책 checkpoint에서도 법칙이 나오는가
- 판정기 몫과 상상 몫은 어떻게 갈리는가
- 어떤 처방이 노출량을 맞춰도 이기는가

**계산 자원**
- 약 93 GPU-시간이다(6.8절 표).
- M1·M3 작업(약 48 GPU-시간)은 대학 GPU 서버에서 돈다.
- M2(Cosmos Policy·LIBERO) 작업(약 45 GPU-시간)은 RunPod에서 돈다. 서버에 sudo와 EGL 렌더링이 없기 때문이다(6.0절). RunPod 비용은 약 $34다.
- E1에서 M2(Cosmos Policy) 곡선이 나오면 go/no-go를 판단한다(6.1절).

**투고**
- arXiv는 12월 중순을 목표로 한다. 경쟁 때문이다.
- 1순위: RSS 2027(1월 말 예상) 또는 ICML 2027(마감 추정 2027-01-22)
- 대안: CoRL 2027, NeurIPS 2027 Datasets & Benchmarks

---

## 1. 문제와 동기

### 1.1 월드모델 기반 정책 평가는 피지컬 AI의 핵심 인프라가 되고 있다
- 실로봇 평가는 비싸고 느리다. 그래서 행동 조건 비디오 월드모델 안에서 정책을 굴려 성공률을 추정하고, checkpoint를 고르고, 정책을 개선한다.
  - Ctrl-World(ICLR 2026)
  - WorldGym
  - Veo 기반 평가(Gemini Robotics)
  - DreamDojo
  - Cosmos Policy(계획 모드에서 후보 행동의 미래와 가치를 예측)
- 보고되는 지표는 대개 실제 성공률과의 상관(Pearson 0.88~0.99)과 순위 보존이다.

### 1.2 "실패를 성공으로 그린다"는 알려졌지만, *어떤* 실패인지는 모른다
- 성공 위주 시연으로 학습한 월드모델은 잘못된 행동에도 성공을 그린다. 2026년에 7편 이상이 이 현상을 보고했다(2.1절).
- 그런데 정책을 개선할수록 실패는 크게 벗어난 실패가 아니라 **아깝게 놓친 실패**가 된다.
- 파일럿이 보인 것은 바로 그 실패를 상상이 가장 못 본다는 것이다.
  - 직접 만든 장면에서는 near-miss를 92.5~99.6% 성공으로 그렸다(실패율 1%).
  - 공개 DINO-WM에서는 성공과 near-miss를 가르는 판별력이 상상 길이와 함께 0.967에서 0.676으로 떨어졌다.
- 그러면 월드모델 점수로 checkpoint를 고를 때, 거의 성공하는 정책을 과대평가하는 체계적 편향이 생긴다.
  - 정책 개선 루프에서 가장 위험한 단계가 거기다.
  - 월드모델 안에서 RL을 하면 reward hacking 경로가 되기도 한다(WMPO, World-VLA-Loop 계열).

### 1.3 원문 에세이와의 연결
- *Neural Networks Are Already Clean*(Cammarata 2026)의 주장
  - 뉴런을 명사가 아니라 **구분**(양극과 음극이 가르는 미세한 대비)으로 봐야 한다.
  - 설명은 near miss로 검증해야 한다.
- near-miss는 결과 공간의 near-tie, 즉 성공과 실패를 가르는 가장 미세한 구분이다.
- 파일럿 R2-11(crushing)의 결과
  - 원문의 "packing은 없다"는 지지됐다.
  - 동시에 넓은 월드모델도 그 미세한 물리(물체 회전, 접촉)를 손실 함수 단계에서 이미 버린다는 것을 보였다. 유닛의 82~96%가 에이전트 운동에 쓰였다.
- 이 연결은 동기와 기전 가설 단락으로만 쓴다. 본론은 정책 평가의 신뢰성이다.

---

## 2. 선행연구와 차별점

### 2.1 가장 가까운 연구

| 연구 | 보인 것 | 우리와의 차이 |
|---|---|---|
| **CureWM**: Xu et al. 2026, *World Models Dream of Success* (arXiv 2610.09134, 10-06, 코드 github.com/jiuyixu25/CureWM). 가장 위협적 | 두 구조 계열의 공개 체크포인트 4개(Cosmos Policy, Ctrl-World)가 실행으로 검증한 실패에 성공을 예측한다. 지표 "optimism"은 실패 반사실 중 성공으로 예측한 비율이다. LIBERO 실패 반사실 484개에서 공식 데이터로 미세조정한 기준선은 80%이고, 성공 시연의 severity grid 반사실로 미세조정하면 평균 38%(30~43%)다. 실패 수·replay·학습 예산을 맞추면 반사실 실패가 새로 모은 on-policy 실패보다 성공-실패 value 차이를 크게 벌린다(0.124 대 0.014). 낙관 감소는 성공-실패 판별력과 함께 봐야 한다고 적었다 | 경계에 가장 가까운 severity 0.2는 실패가 2개 이하라 분석에서 뺐다. 본문에 "policy ranking is untested"라고 적었다(NOVELTY_R2 확인). 판별력도 실패 전체로만 봤다. 실패 거리별 곡선과 성공 vs near-miss 해상도, 정책 수준 법칙, 원천 분해, 평가 쪽 처방이 없다 |
| MiraBench (arXiv 2605.29360) | optimism bias 탐지를 포함한 action-conditioned 신뢰성 벤치마크. 인간 판정 16,000개 | severity가 하나(s = 0.5)뿐이고, near/far 비교와 정책 순위가 없다 |
| World-VLA-Loop (arXiv 2602.06508) | "minor action errors"로 인한 실패를 시뮬레이션하지 못한다. near-success 데이터(SANS)로 고친다 | 정성적이다. 실패 거리 곡선과 정책 평가 편향이 없다 |
| HF-WM: Tu et al. 2026 (arXiv 2610.13522, 10-09) | 성공 시연이 지배해 실패 동역학이 과소 대표된다. 성공 시연에서 만든 교란 행동에 GRPO 후학습을 해 실패 동역학을 배운다. 정책 평가에 쓸 수 있다고 주장한다 | 처방 쪽 연구다. 초록 기준으로 경계 근처 측정과 정책 순위가 없다 |
| FACT (2608.10232), DreamTrue (2610.12468), dWorldEval (2604.22152), WorldSync (2608.24885) | 성공 편향 환각과 실패 데이터 처방 | 같음 |
| FARM: Pei et al. 2026 (arXiv 2609.11445) | 고정 월드모델(VLA-JEPA)의 내부 예측 상태에서 실패 정보를 읽는다(AUROC 85.7) | 실제 실행 중 모니터링이다. 상상 속 판정이 아니다. 우리 평가 쪽 처방의 기준선 후보다 |
| *Completion Aware Guidance*(NeurIPS 2026) | 월드 액션 모델의 "task-incomplete imagination"(완료 전이를 빠뜨린 상상)을 79%에서 40%로 줄이는 학습 없는 guidance | 정책이 상상한 미래의 문제다. 평가용 월드모델의 경계 해상도 측정은 아니다 |
| 정책 평가용 월드모델: WorldGym, WorldEval, Ctrl-World, Veo 시뮬레이터, RoboWorld, GigaWorld, DreamDojo | 실제 성공률과 상관이 높다(0.88~0.99) | near-miss로 층화한 분석이 없다. Veo 시뮬레이터는 절대 성공률을 대체로 과소 추정했다는 요약이 검색됐다(원문 미확인). "낙관이 보편적"이라는 주장을 피해야 할 근거다 |
| 평가 지표와 판정기: SIMPLER의 MMRV, FailBench (arXiv 2609.03611) | MMRV는 성공률이 비슷한 정책 사이의 순위 위반에 관대하도록 설계됐다. VLM 실패 판정기는 모호할 때 성공으로 기운다 | 우리 핵심 사례(52.6% 대 48.2%가 상상 98.5% 대 58%)는 MMRV로 거의 벌점을 받지 않는다 |
| 클래스 불균형 이론: LDAM (Cao et al. 2019) | 불균형하면 결정 경계가 소수 클래스 쪽으로 밀린다 | 경계 팽창(r* 이동)은 이 고전 현상의 생성 월드모델판이다. 서론에서 먼저 밝힌다 |

### 2.2 남은 공백 (NOVELTY_R2 §2.3)
1. 경계로부터의 거리에 따른 P(상상 성공 | 실제 실패) 곡선. 경계 근처를 포함해야 한다.
2. 정책의 near-miss 비중이 평가 편향을 정한다는 정책 수준 법칙, 그리고 실제 성공률이 같은 정책의 순위 역전
3. 판정기 몫과 상상 몫의 분리
4. 경계 해상도(성공 vs near-miss 판별력) 지표와, 실패 기하를 반영한 순위 지표
5. 노출량을 맞춘 처방 비교와, 재학습 없는 평가 쪽 처방
6. 공개 latent 월드모델(DINO-WM)과 정책 평가용 비디오 월드모델(Cosmos Policy)에서의 검증

### 2.3 서술 원칙
- 첫 단락에서 성공 편향 환각을 MiraBench, CureWM, FACT, World-VLA-Loop, dWorldEval, HF-WM이 보였다고 인정한다.
- 기여는 "얼마나 먼 실패까지 성공으로 보이는가"의 곡선과, 그것이 정책 평가와 checkpoint 선택에 주는 결과로 정의한다.
- **쓰지 않을 주장**
  - "월드모델은 낙관적이다"(일반 명제). 공개 DINO-WM은 순 편향이 비관이었다(R2-12). Veo는 과소 추정했다는 요약이 있다. 대신 "실패 기하에 따른 조건부 편향"과 "경계 해상도 손실"로 쓴다.
  - "통합(consolidation) 지연이 기전이다." MLP에서만 지지됐고, ViT 예측기에서는 아니었다(R2-9, ρ = −0.43).
  - "최초 발견." 위 연구들이 있다.

---

## 3. 연구 질문과 가설 (사전등록 대상)

| RQ | 질문 | 가설(사전등록) |
|---|---|---|
| RQ1 곡선 | 상상이 실패를 성공으로 판정할 확률은 실패 거리 r과 어떤 관계인가? | H1a: 모델마다 P(Ŝ = 1 \| 실제 실패, r)가 r에 대해 단조 감소한다. near-miss 구간(1 ≤ r < 2)이 far-miss(r ≥ 3)의 3배 이상이고 0.3 이상인 모델이 3개 중 2개 이상이다. H1b: 성공 vs near-miss AUROC는 상상에서 실제 프레임보다 0.05 이상 낮다. 성공 vs far-miss AUROC의 하락은 0.02 이하다(near-miss 특이적 손실). 이 차이는 상상 길이와 함께 커진다 |
| RQ2 정책 법칙 | 정책 평가의 편향은 실패의 기하를 따르는가? | H2a: 학습된 정책 checkpoint 집합에서 상상이 만든 과대 추정 o_π^imag가 near-miss 비중 n_π와 Spearman 0.6 이상이다(모델 2개 이상). H2b: 실제 성공률 차이 ≤ 5%p인 정책 쌍에서 상상 성공률 차이가 15%p 이상인 쌍이 하나 이상 있다. 그런 정책 집합 안의 Kendall τ(상상 대 실제)는 0.5 이하다. H2c: 같은 조건에서 전체 정책 Pearson은 0.8 이상이고 MMRV는 0.05 이하로 "좋게" 보인다(표준 지표의 맹점) |
| RQ3 원천 | 오판은 판정기 때문인가 상상 때문인가? | H3: 2×2(판정기 × 실제 / 상상 프레임) 분해에서 near-miss 오판의 과반이 상상 몫이다(모델 2개 이상). VLM 판정기는 실제 프레임에서도 자체 낙관을 더한다(보고) |
| RQ4 기전 | 무엇이 양방향 흐림을 낙관으로 바꾸나? 정보는 어디서 사라지나? | H4a(인과): 같은 월드모델을 성공 편중 데이터로 미세조정하면 near-miss의 상상 성공이 늘고, 실제 성공의 상상 실패는 준다. 실패율 p에 대한 용량-반응이 있다. H4b(위치): 실제 프레임 특징에는 near-miss 정보가 남아 있다(정보 상한 AUROC ≥ 0.95). 상상 특징에서 사라진다(R2-12 D3 재현) |
| RQ5 처방 | 무엇이 경계를 되살리나? | H5a(데이터): 실패 노출량을 맞추면 경계 근처 반사실(near-miss 위주 severity grid) 증강이 재가중과 CureWM식 전 범위 반사실보다 near-miss 오판을 더 줄인다(설정 2개 이상). 성공 오판 증가는 5%p 이하다. H5b(평가 쪽, 재학습 없음): 상상 불확실성(앙상블 / MC dropout)으로 경계 근처 상상 성공을 할인하거나 보류하면 정책 수준 편향과 순위 오류가 준다 |
| RQ6 결과 | 이 편향이 checkpoint 선택을 얼마나 망치나? | H6: 월드모델 점수로 고른 checkpoint의 regret(실제 최선 대비)이 near-miss 비중이 높은 checkpoint에서 생긴다. 처방은 regret을 줄인다 |

---

## 4. 방법

### 4.1 정의
- **실패 거리 r**: 과제마다 미리 정한 연속 오차 e를 그 과제의 성공 문턱 τ로 나눈 값이다(r = e / τ).
  - 성공은 r < 1이다.
  - 과제별 e의 정의는 사전등록한다.
    - PushT: 평가기의 위치·자세 오차. agent 위치를 포함한 원 정의와 물체 자세만의 정의 두 가지
    - LIBERO: 물체와 목표 영역 경계의 거리, 또는 관절 각 문턱까지의 거리
    - 통제 장면: 정렬 오차 / 반폭
- **구간(보조)**: 성공 r < 1, near-miss 1 ≤ r < 2, mid 2 ≤ r < 3, far-miss r ≥ 3(R2-12와 같음)
  - 주 분석은 구간이 아니라 연속 곡선이다(4.3절).
  - R2-9의 정렬비 [1, 1.2)는 통제 장면 보조 구간으로 함께 보고한다.
- **상상 판정 Ŝ**: 같은 초기 관측과 같은 행동열(또는 같은 정책)로 월드모델을 굴린 뒤, 예측 최종 관측을 판정기 j로 판정한다.
- **판정기 j**(모델마다 해당하는 것)
  - (a) **oracle 판별기**: 실제 최종 프레임의 특징 + 목표로 학습한 작은 분류기. 실제 프레임 정확도 0.95 이상을 요구한다(R2-12 (a)).
  - (b) 목표 latent 거리 문턱(DINO-WM 계획 비용의 시각 항)
  - (c) 월드모델 자체의 value head(Cosmos Policy)
  - (d) VLM 판정기(공개 VLM, 고정 프롬프트, 목표 문장 포함)
- **평가 모드**
  - **O(open-loop 재생, 주 분석)**: 실제 환경에서 실행한 정책의 행동열을 상상으로 재생한다. 정책과 월드모델의 상호작용을 배제하고 결과 예측만 본다.
  - **C(closed-loop 상상, 보조)**: 정책이 상상된 관측을 보고 행동한다. 실제 정책 평가 절차이며, LIBERO에서만 쓴다.

### 4.2 모델과 과제
| 기호 | 월드모델 | 과제·데이터 | 참값 | 역할 |
|---|---|---|---|---|
| M1 | DINO-WM PushT 공개 체크포인트(OSF) | PushT, val 분할 시작·목표 쌍 | 시뮬레이터 재생 | 공개 latent 월드모델(R2-12의 확장) |
| M2 | **Cosmos Policy LIBERO**(`nvidia/Cosmos-Policy-LIBERO-Predict2-2B`, Cosmos-Predict2 2B 기반. 행동·미래 영상·value를 출력하고 추론에 VRAM 약 6.8GB) | LIBERO의 연속 성공 여유를 정의할 수 있는 과제 4~8개(사전 선정) | LIBERO 시뮬레이터 | 정책 평가에 쓰이는 비디오 월드모델. 반사실 질의는 CureWM 코드에서 재사용 |
| M3 | 통제 모델: DINO-WM식 latent 예측기(R2-9) | 2D 집어 들기 렌더 장면, 학습 데이터 실패율 p ∈ {0.3, 0.1, 0.03} | 결정론적 시뮬레이터 | 데이터 균형의 인과 조작(RQ4), 처방(RQ5) |
| (선택) M4 | Ctrl-World(코드 github.com/Robert-gyj/Ctrl-World. 공개 체크포인트와 학습 데이터는 프로젝트 페이지에서 확인되지 않음) | 체크포인트의 학습 도메인에 맞춤 | 시뮬레이터 도메인이면 시뮬레이터, 아니면 없음 | 체크포인트를 E0 안에 돌릴 수 있으면 순위 실험만 보조로 넣음 |

- 실행 장소: M1·M3은 대학 GPU 서버, M2와 M4는 RunPod다(6.0절).

### 4.3 지표
- **실패 거리 곡선**
  - P(Ŝ = 1 | r)를 r의 로그에 대한 로지스틱(모델 × 판정기 × horizon 상호작용)으로 맞춘다.
  - 요약값
    - r₅₀: 상상 성공 확률이 0.5로 떨어지는 실패 거리
    - 흐림 폭 Δr₁₀ = r₁₀(상상) − r₁₀(실제 프레임): 상상이 경계를 얼마나 바깥으로 밀어내는가
  - 구간 확률(near / mid / far)을 함께 보고한다.
- **경계 해상도**
  - AUROC(성공 대 near-miss)와 AUROC(성공 대 far-miss)를 실제 / 상상 특징에서 잰다.
  - 정보 상한: 상상 특징으로 직접 학습한 판별기의 AUROC(R2-12 D3)
- **정책 수준 지표**
  - 실제 성공률 s_π
  - 상상 성공률 ŝ_π
  - 과대 추정 o_π = ŝ_π − s_π와 그 분해: 상상 몫 o_π^imag = ŝ_π − ŝ_π^real-judge, 판정기 몫 = ŝ_π^real-judge − s_π
  - near-miss 비중 n_π
- **순위 지표**
  - Spearman, Kendall τ(전체, 그리고 실제 성공률이 5%p 이내인 정책 집합 안)
  - Pearson, MMRV(SIMPLER 정의)
  - 제안 지표: **실패 기하 가중 순위 오류**. 실제 성공률이 비슷한 쌍의 순위 위반에, 두 정책의 near-miss 비중 차이로 가중을 준다. 정의는 사전등록한다.
- **checkpoint 선택 regret**: 월드모델 점수 1위 checkpoint의 실제 성공률과 실제 최선 checkpoint의 실제 성공률의 차이

---

## 5. 지금까지의 증거 (파일럿, 모두 사전등록)

| 파일럿 | 설정 | 판정 | 핵심 수치 |
|---|---|---|---|
| [R2-4](pilots/R2_4_optimism_bias_mechanism/REPORT.md) | 2D 집어 들기, 상태 예측기, 실패율 p | 없음(기전 실패, 현상 재현) | p=0.01에서 낙관 편향 30.6%(p=0.3의 17.8배). 결정 경계 r* 1.02 → 1.39. 경계 바로 바깥 실패의 99.6%를 성공으로. 전용 관측 단서가 있는 실패는 0.25% 빈도에서도 편향 0(one-hot 입력) |
| [R2-6](pilots/R2_6_optimism_consolidation/REPORT.md) | 상태 + DINOv2 시각 버전 | 조건부(사후 점검 반영) | 실패율 1%에서 near-miss 99.6%(상태) / 92.5%(시각) 대 far-miss 5.4% / 6.0%. r* 이동 +0.37 / +0.22. 같은 데이터로 학습만 4배 늘리면 편향 −43%(고유 실패 1k) / −77%(3k)(상태). 고유 실패 200개면 −7.5%뿐. 시각 버전에서는 단서가 있는 실패도 1×에서 99.9% 성공으로 그림(입력 표현이 중요). 재가중·near-miss 재샘플링은 학습을 늘리거나 시각 입력이면 오히려 악화(시각 0.48 → 0.71) |
| **[R2-9](pilots/R2_9_policy_eval_optimism/REPORT.md)** | DINO-WM식 latent 비디오 예측기(4단계 자기회귀) + 고정 성공 판별기, 학습 데이터 실패율 3%, 정책 6개 | 조건부(기전만) | 과대 추정 +16.9%p. **실제 52.6% / 48.2%인 near-failer / far-failer가 상상 98.5% / 58.0%.** near-miss 비중-과대 추정 r = 0.95. 판별기의 실제 프레임 정확도 1.000. 4배 학습 −52%. 처방: 대비 쌍 −53%, 재가중 −64%(노출량 미맞춤) |
| [R2-11](pilots/R2_11_crushing_world_model/REPORT.md) | 시각 월드모델 crushing | 없음 | packing 없음. 넓은 모델도 유닛의 82~96%를 에이전트에 씀. 회전·접촉 정보는 학습 중 소실. 거칠어짐은 손실 함수 단계에서 이미 일어남(경계 정보 손실의 기전 후보) |
| **[R2-12](pilots/R2_12_dinowm_nearmiss_optimism/REPORT.md)** | **공개 DINO-WM PushT**, 정책 8개 × 400 episode, H=5 | 있음(약함, 판정기 (b)에서만) | 판정기 (b) P(Ŝ \| near) 0.383 대 far 0.002(178배), (a) 0.205. 성공 vs near-miss AUROC: 실제 0.967, 상상 0.880, H10 0.676. 성공 vs far-miss는 0.996(H10 0.944). 양방향 흐림: near-miss는 +6~10%p 성공 쪽, 실제 성공은 −9~12.5%p 실패 쪽, 순 편향 −3~−7%p. 상상이 만든 과대 추정은 near-miss 비중을 따름(ρ 0.76~0.91, 사후). 판정기를 상상에 맞춰 보정하면 R2-9 모양(ρ 0.97). 한계: 판정기 (b)는 실제 프레임에서도 near-miss의 28.3%를 성공으로 봄. near-miss의 85%가 agent 위치 오차 |

**확립된 것**
- 경계 근처 정보 손실은 통제 장면과 공개 DINO-WM 모두에서 나타난다. 상상 길이와 함께 커지고, far-miss에는 생기지 않는다.
- 성공 위주 데이터에서는 그것이 강한 낙관과 정책 순위 역전으로 나타난다(R2-9).

**미확립(본 실험의 대상)**
- 정책 평가용 비디오 월드모델(M2)
- 학습된 정책 checkpoint(손으로 설계한 교란 정책이 아니라)
- 판정기와 상상의 분리(VLM 포함)
- 데이터 균형의 인과
- 노출량을 맞춘 처방 비교와 평가 쪽 처방

---

## 6. 본 실험 계획

### 6.0 공통 프로토콜 (사전등록 대상)

**실행 장소**(모델 단위로 고정)
- **M1·M3: 대학 GPU 서버.** 맥북에서 Tailscale과 Windows 중계를 거쳐 접속한다(ssh 별칭 `kiis-mvf-gpu`).
  - 장비: Ubuntu 22.04 컨테이너, RTX A5000 24GB × 2, 64스레드, RAM 503GiB. CPU와 디스크는 다른 사용자와 공유한다.
  - DINO-WM PushT는 pygame·pymunk로, M3 장면은 torch로 그려 OpenGL이 필요 없다. 서버의 기존 venv에서 pygame·pymunk·opencv가 동작하는 것을 확인했다(2026-10-21).
  - 환경은 사용자 공간 uv venv로 만든다(서버에 Docker·sudo·conda 없음).
- **M2(와 선택 M4): RunPod.**
  - 서버에는 sudo와 EGL·OpenGL 라이브러리가 없다. 그래서 Cosmos Policy(공식 설치는 Docker 기반)와 LIBERO(EGL 렌더링)에 필요한 시스템 라이브러리를 깔 수 없다.
  - RunPod pod는 root라 apt와 uv로 같은 환경을 만들 수 있다. ECM에서 RunPod pod의 LIBERO가 `MUJOCO_GL=egl`로 동작했다.
- **섞지 않는다.** 다음은 같은 장소, 같은 GPU 모델에서 돈다. 결과는 두 장소 모두 맥북으로 모아 분석한다.
  - 같은 모델의 모든 조건(판정기, horizon, 처방)
  - 같은 정책의 실제 rollout과 상상 rollout
- 서버 작업과 RunPod 작업은 E0부터 나란히 진행한다.

**정책 집합**
- **M1(PushT)**
  - Diffusion Policy(Chi et al. 2023)를 표준 PushT 시연으로 seed 3개 학습하고, 학습 단계 8개에서 checkpoint를 저장한다(24개).
  - 보조로 R2-12의 교란 정책 8개를 그대로 둔다(통제 대비용).
- **M2(LIBERO)**
  - 선정 과제마다 행동복제 정책(작은 Diffusion Policy)을 학습해 4개 단계 checkpoint를 둔다.
  - 공개 강한 정책(openpi π0.5 LIBERO, OpenVLA-OFT)을 기준점으로 함께 둔다.
  - 보조로 R2-12식 교란 계열(지속 offset → near-miss, iid → far-miss)을 같은 성공률로 보정해 둔다.
- **M3**: R2-9의 정책 6개(near-failer / far-failer 대조 포함)

**반사실 실패 생성(곡선용)**
- 성공 궤적의 행동에 severity grid 교란을 더한 반사실을 시뮬레이터로 실행해 r을 얻는다(CureWM 방식).
- 단, grid를 경계 근처에 촘촘하게 둔다.
  - severity를 r ∈ [0.8, 3]에 균등하게 맞추도록 보정 sweep으로 정한다.
  - 모델당 near-miss 궤적을 1,000개 이상 확보한다.

**분할**
- 판정기 학습 데이터, 보정 데이터, 평가 데이터의 시작 상태가 겹치지 않게 한다.
- 월드모델 학습 분할과의 겹침도 기록한다. R2-12의 train 분할 점검을 이어받는다.

**episode 수**
- 곡선: 모델 × 판정기마다 궤적 3,000개 이상
- 정책 평가: checkpoint당 과제마다 100 episode(PushT) / 50 episode(LIBERO 과제당)

**판정기 기준**
- oracle 판별기는 실제 프레임 held-out 정확도 0.95 이상을 요구한다. 못 미치면 판정에 쓰지 않는다.
- 판정기마다 "지각 바닥", 즉 실제 프레임에서의 near-miss 오판을 먼저 보고한다(R2-12 지각 단서 규칙).

### 6.1 실험 1 — 실패 거리 곡선 (RQ1)
- **대상**: M1(H ∈ {3, 5, 7, 10}), M2(행동 청크 1~4개 상상), M3(p ∈ {0.3, 0.1, 0.03})
- **측정**: P(Ŝ = 1 | r) 곡선, r₅₀, 흐림 폭 Δr₁₀, 구간 확률, 경계 해상도 AUROC(실제 / 상상 / 상상 특징 학습 판별기)
- **판정(H1)**
  - H1a: 모델 3개 중 2개 이상에서 near ≥ 0.3이고 near / far ≥ 3이다. 판정기 (a) 기준을 우선하고 (c)와 (d)를 함께 보고한다.
  - H1b: 같은 2개 이상에서 Δ AUROC(성공 vs near) ≥ 0.05이고 Δ AUROC(성공 vs far) ≤ 0.02다. horizon 기울기 95% CI가 음수다.
- **go/no-go(E1의 M2 결과로 판단)**
  - M2에서 H1a 또는 H1b가 성립하면 go다.
  - M2에서 둘 다 실패하면 주장을 "latent 월드모델(M1·M3)의 경계 해상도"로 좁히고, 워크숍이나 RA-L 경로로 바꾼다.

### 6.2 실험 2 — 정책 수준 법칙과 순위 역전 (RQ2)
- **대상**: M1 checkpoint 24개와 교란 정책 8개, M2 checkpoint와 기준 정책, M3 정책 6개
- **측정**
  - o_π와 그 분해(상상 몫 / 판정기 몫), n_π
  - 순위 지표 전체(4.3절)
  - 모드 O를 주로 쓰고, M2는 모드 C도 함께 잰다.
- **분석**
  - o_π^imag ~ n_π 회귀. 정책 수준 cluster bootstrap(episode 재표집)을 쓴다.
  - 실제 성공률 5%p 이내 쌍의 상상 차이 분포
  - 표준 지표(Pearson, MMRV)와 제안 지표의 비교
- **판정(H2)**
  - H2a: Spearman(n_π, o_π^imag) ≥ 0.6(모델 2개 이상)
  - H2b: 조건을 만족하는 쌍이 하나 이상 있다. 비슷한 성공률 집합 안의 Kendall τ ≤ 0.5다.
  - H2c: 같은 모델에서 전체 Pearson ≥ 0.8이고 MMRV ≤ 0.05다.

### 6.3 실험 3 — 원천 분해 (RQ3)
- **2×2(또는 4×2)**: 판정기 {(a) oracle, (c) value head, (d) VLM, (b) latent 거리} × 입력 {실제 최종 프레임, 상상 최종 프레임}
- **보고**
  - near-miss 오판을 지각 바닥(실제 프레임)과 상상 증분으로 나눈다.
  - FailBench 결과(VLM의 성공 쪽 편향)와 연결한다.
- **판정(H3)**: 상상 증분이 near-miss 오판의 50%를 넘는 모델이 2개 이상이다.

### 6.4 실험 4 — 기전 (RQ4)
- **4a 데이터 균형의 인과 시험**
  - M1(DINO-WM PushT)을 성공 편중 데이터(목표 도달 구간 97%)와 균형 데이터로 같은 step만큼 짧게 미세조정한다. R2-12 다음 단계 2를 잇는다.
  - M3는 p를 바꿔 처음부터 학습한다.
  - 같은 정책 집합에서 "양방향 흐림"이 "낙관"으로 바뀌는지, 그리고 p에 대한 용량-반응을 본다.
- **4b 정보가 사라지는 위치**
  - 실제 프레임 특징의 정보 상한을 잰다(near-miss 판별 AUROC).
  - 상상 단계별 AUROC 감소를 잰다.
  - 예측기 표현의 해상도를 sweep한다(토큰 64 / 256, PCA 64 / 384, 예측기 크기). R2-9 다음 단계 3이다.
- **4c 손실 단계 coarsening(R2-11 연결)**
  - 학습 손실에서 경계 관련 변수(물체 자세, 접촉)가 차지하는 분산 몫을 잰다.
  - 그 몫을 키우는 **경계 가중 손실**(물체·접촉 근처 patch 가중)을 처방 후보로 넘긴다.
- **판정(H4)**
  - H4a: 성공 편중 미세조정에서 near-miss 상상 성공 증가와 실제 성공 상상 실패 감소가 모두 짝 CI로 유의하다(M1 또는 M3).
  - H4b: 실제 프레임 정보 상한 ≥ 0.95이고, 상상에서 0.05 이상 떨어진다.

### 6.5 실험 5 — 처방 (RQ5)
- **데이터 쪽(노출량 맞춤)**: 같은 실패 수와 같은 batch 내 실패 비율로 비교한다. R2-9가 노출량을 맞추지 못한 한계를 고친다.
  - (i) 균일 재가중
  - (ii) near-miss 반사실 증강(경계 근처 severity grid)
  - (iii) 대비 쌍(같은 시작 상태에서 경계 양쪽 행동)
  - (iv) CureWM식 반사실(severity 전 범위, 주 기준선)
  - (v) 경계 가중 손실(4c)
- **평가 쪽(재학습 없음)**
  - 앙상블 또는 MC dropout(DINO-WM 예측기 dropout 0.1)의 상상 불확실성으로 경계 근처 상상 성공에 할인을 준다. 또는 판정을 보류한다.
  - 실제 실행 중 실패 신호를 읽는 FARM류 readout을 상상에 적용한 기준선을 둔다.
- **판정(H5)**
  - H5a: (ii)가 (i)과 (iv)보다 near-miss 오판을 더 줄인다(설정 2개 이상, 짝 CI). 성공 오판 증가는 5%p 이하다. (iii)은 R2-9에서 짝 자체의 효과가 없었으므로(cp ≈ nm) 보조로 보고한다.
  - H5b: 평가 쪽 처방이 정책 수준 |o_π| 평균과 실패 기하 가중 순위 오류를 줄인다(M1·M2 중 1개 이상).

### 6.6 실험 6 — checkpoint 선택의 결과 (RQ6)
- M1 checkpoint 24개와 M2 checkpoint로 "월드모델 점수 1위 고르기"를 시뮬레이션한다.
- regret, 고른 checkpoint의 near-miss 비중, 처방 전후를 비교한다.
- **선택**: 월드모델 안에서 정책을 개선하는 작은 루프(PushT, reward = 상상 성공)를 돌려, 정책이 near-miss 쪽으로 수렴하는지(reward hacking) 본다.
- **판정(H6)**: 처방 없는 선택의 regret이 0보다 유의하게 크다. 처방이 그것을 줄인다(짝 bootstrap).

### 6.7 통계와 사전등록
- **곡선**
  - 로지스틱 GLMM(시작 상태 무작위 효과)으로 맞춘다.
  - r₅₀와 Δr₁₀의 CI는 시작 상태 cluster bootstrap으로 낸다.
- **정책 수준**: episode cluster bootstrap(10,000회) 95% CI
- **다중 비교**
  - 1차 가설군(H1a, H1b, H2a)에 Holm 보정을 한다.
  - 나머지는 탐색으로 표시한다.
- **검정력**
  - near-miss 궤적 1,000개면 P(Ŝ \| near) ≈ 0.38의 95% 구간 반폭은 독립 가정에서 약 ±0.03이다.
  - 시작 상태 cluster 때문에 실제 구간은 더 넓다. R2-12는 val 궤적 21개에서 시작해 [0.326, 0.431]이었다. 본 실험의 학습된 정책은 시뮬레이터의 무작위 시작 상태에서 돌리므로 시작 상태 수를 episode 수만큼 늘릴 수 있다.
  - 정책 법칙은 정책 수가 검정력을 정한다(M1 32개). 참 ρ = 0.6이면 정책 32개로 ρ = 0을 α = 0.05에서 기각할 검정력은 약 0.95다(Fisher z 근사).
- **사전등록 문서**: 실험 1 본 실행 전에 `DESIGN_main.md`로 고정한다. 과제별 e와 τ 정의, 판정기 선택 규칙, 정책 집합, seed 범위, 처방의 노출량 정의를 적는다.
- **보고 원칙**
  - 지각 바닥을 항상 함께 보고한다.
  - 판정기별 결과를 숨기지 않는다.
  - 판정 실패도 모두 보고한다.

### 6.8 계산 예산과 실행 순서

**처리량 실측과 추정**
- M1: RTX 3090 pod 36분에 판별기 학습(약 11분)과 평가 집합 3개(본 집합 3,200 궤적)의 상상·분석을 끝냈다(R2-12, $0.31).
- M3: RTX A5000 pod 86분에 모델 24개 학습·평가를 끝냈다(R2-9, $0.40).
- M2: Cosmos Policy 2B는 실측이 없다. 예측 1회 약 1~2초로 가정하고 E0 스모크에서 다시 잰다.
- M1·M3의 파일럿 처리량은 RunPod GPU 기준이다. 서버(A5000 × 2)에서 E0에 다시 잰다. R2-9는 A5000에서 돌았다.

| 실험 | 규모 | 서버 GPU-시간(M1·M3) | RunPod GPU-시간(M2) | RunPod 비용(추정) |
|---|---|---|---|---|
| E0 설치·스모크·정책 학습 | 서버: DINO-WM PushT·M3 환경, PushT DP 3 seed. RunPod: Cosmos Policy·CureWM 코드, LIBERO 과제·e 정의, LIBERO DP | 7 | 8 | $6 |
| E1 곡선 | 서버: M1 × horizon 4, M3 p 3개. RunPod: M2 반사실 약 3,000 | 14 | 16 | $12 |
| E2 정책 법칙 | 서버: M1 32 정책 × 100 ep. RunPod: M2 checkpoint × 과제 × 50 ep(모드 O + C) | 6 | 14 | $10 |
| E3 원천 분해 | 판정기 4종 × 2입력. VLM은 양쪽, value head는 M2 | 2 | 4 | $3 |
| E4 기전 | DINO-WM 미세조정 2조건 + M3 sweep + 해상도 sweep | 10 | 0 | $0 |
| E5 처방 | 서버: 데이터 쪽 5종 × 2설정(M1·M3), 평가 쪽(M1). RunPod: M2 평가 쪽(선택) | 6 | 2 | $2 |
| E6 선택·개선 루프 | 서버: M1 선택·소형 RL. RunPod: M2 선택(E2 결과 재사용) | 3 | 1 | $1 |
| **합계** | | **약 48** | **약 45** | **약 $34** |

- 장소별 배분은 추정이다. E0의 처리량 측정으로 다시 정한다.

- LIBERO 시뮬레이터 재생은 CPU를 많이 쓴다. M2 pod는 CPU 할당이 큰 것(작업자 20개 이상)을 고른다. 사용자는 ECM에서 RunPod pod로 LIBERO·openpi를 돌려 봤다.

**실행 순서**(앞 단계의 결과가 다음 단계의 입력이다. 서버와 RunPod 작업은 나란히 진행한다)

| 순서 | 할 일 | 장소 | 산출물 |
|---|---|---|---|
| 1 | E0: 서버 환경(uv venv)과 처리량 측정, DINO-WM PushT·M3 재현, PushT DP 학습 시작 | 서버 | 환경 점검표, 재현표 |
| 1 | E0: Cosmos Policy 반사실 질의 재현(CureWM의 optimism 재현이 점검 기준), LIBERO 과제·e 정의, LIBERO DP 학습 | RunPod | 스모크 보고 |
| 2 | `DESIGN_main.md` 고정 | 맥북 | 사전등록 |
| 3 | E1 | 서버(M1·M3), RunPod(M2) | 곡선 그림, **go/no-go**(M2 곡선으로 판단) |
| 4 | E2 | 서버(M1), RunPod(M2) | 정책 법칙·순위 표 |
| 5 | E3, E4 | 서버(M1·M3), RunPod(M2 판정기) | 원천 분해, 기전(인과 시험 포함) |
| 6 | E5 | 서버(데이터 쪽), RunPod(M2 평가 쪽, 선택) | 처방 표 |
| 7 | E6 | 서버(M1), RunPod(M2) | 선택 regret |
| 8 | 집필 | — | arXiv, 투고본 |

### 6.9 재사용할 코드

| 용도 | 위치 |
|---|---|
| DINO-WM PushT 평가 집합, 교란 정책 보정, 판정기 (a)·(b) 학습, 곡선·정책 분석 | [pilots/R2_12_dinowm_nearmiss_optimism/code/](pilots/R2_12_dinowm_nearmiss_optimism/code/): `calib12.py`, `clfset12.py`, `judges12.py`, `main12.py`, `analyze12.py`, `posthoc12.py`, `setup12.sh` |
| 통제 렌더 장면, DINO-WM식 예측기, 정책 6개, 정책 평가 | [pilots/R2_9_policy_eval_optimism/code/](pilots/R2_9_policy_eval_optimism/code/): `world.py`, `wm.py`, `evaluate.py`, `summarize.py` |
| 실패율 p 조작, 경계 r*, n50 분석, 상태·시각 버전 | [pilots/R2_6_optimism_consolidation/code/](pilots/R2_6_optimism_consolidation/code/), [pilots/R2_4_optimism_bias_mechanism/code/](pilots/R2_4_optimism_bias_mechanism/code/) |
| 변수별 분산 몫, probe, 손실 단계 coarsening 분석 | [pilots/R2_11_crushing_world_model/code/](pilots/R2_11_crushing_world_model/code/): `posthoc_probe.py`, `posthoc_fine.py` |
| Cosmos Policy LIBERO | github.com/nvlabs/cosmos-policy(LIBERO.md, Docker), HF `nvidia/Cosmos-Policy-LIBERO-Predict2-2B` |
| 반사실 severity grid, optimism 지표 | github.com/jiuyixu25/CureWM |
| 서버 실행(M1·M3, 새로 만듦) | ssh 별칭 `kiis-mvf-gpu`. 코드와 결과는 rsync로 옮기고, 작업은 tmux에서 작업 목록 기반으로 돌려 중단 뒤에도 이어 간다. 파일럿 설치 스크립트(`setup12.sh` 등)는 RunPod용(root·apt)이라 사용자 공간 uv 설치로 고친다 |
| RunPod(M2) | [../runpod_infra/](../runpod_infra/). `budget.py`의 START_BALANCE·LIMIT·HARD_DEADLINE과 장부, `common.sh`의 pod 접두어를 새로 정한다 |

---

## 7. 리스크와 대응

| 리스크 | 현재 근거 | 대응 |
|---|---|---|
| 경쟁: CureWM 저자군이 정책 순위를 다음 과제로 남김 | 2026-10-06 공개, 같은 저자군이 PreDE도 냄 | 12월 중순 arXiv. 범위를 실패 거리 곡선, 정책 수준 법칙, 원천 분해로 좁게 잡아 빨리 낸다 |
| M2에서 Cosmos Policy 반사실 질의가 막힘 | README의 빠른 시작은 `get_action`(행동·미래 영상·value 반환)만 보여 준다. 계획(best-of-N)은 ALOHA 쪽에만 언급되고, 계획 코드 공개 여부는 확인하지 못했다 | CureWM이 같은 공개 체크포인트에 반사실 행동을 넣어 평가했으므로 그 코드부터 본다. E0 안에 풀리지 않으면 LIBERO 렌더로 학습한 통제 모델(M3의 LIBERO판)을 RunPod에서 만들고, M2는 value head 판정만 쓴다 |
| 공개 모델에서 효과가 약함 | R2-12는 판정기 (b)에서만 통과했고, 크기는 R2-9의 1/4~1/3 | 주 지표를 경계 해상도 AUROC와 흐림 폭으로 둔다(0.967 → 0.676은 강하다). 데이터 균형 인과 시험(E4a)으로 효과가 커지는 조건을 보인다 |
| 학습된 checkpoint의 near-miss 비중이 다양하지 않음 | 미측정 | 학습 단계 8개 × seed 3개. 부족하면 교란 정책 계열을 보조 정책 집합으로 쓴다(R2-12에서 0%~47.5%) |
| 판정기 confound(지각 흐림) | R2-12 (b)는 실제 프레임에서도 near 28.3%를 성공으로 봄 | oracle 판별기 기준(정확도 0.95 이상)을 우선하고, 지각 바닥을 항상 분리 보고(E3) |
| PushT 성공 조건이 agent 위치를 포함 | near-miss의 85%가 agent 위치 오차 | 물체 자세만의 성공 정의로 반복하고, 두 정의를 모두 보고 |
| 결정론적 실패 가정 | 모든 파일럿이 결정론적 실패 | LIBERO에 마찰·질량 잡음을 넣은 확률적 실패를 보조로 시험 |
| 낙관이 보편적이지 않음(Veo 과소 추정) | 문헌 요약 | 주장을 "실패 기하에 따른 조건부 편향"과 "경계 해상도"로 쓴다. 비관 쪽 결과도 같은 틀(양방향 흐림)로 설명 |
| 예산 | RunPod 추정 약 $34(M2만) | 우선순위 E1 > E2 > E3 > E4a > E5 > E6. M1·M3은 서버, M2는 RunPod |
| 두 장소의 결과가 섞임 | 서버(A5000)와 RunPod의 GPU가 다름 | 모델 단위로 장소를 고정하고, 한 비교 안에서 장소를 섞지 않음(6.0) |
| 서버 공유 자원 | CPU·디스크를 다른 사용자와 공유(디스크 86% 사용) | 큰 데이터(LIBERO)는 서버에 두지 않음. 결과는 맥북으로 수시 동기화 |

---

## 8. 논문 구성과 투고 전략

**제목 후보**
- *Near-Miss Blindness: World-Model Imagination Loses the Outcome Boundary*
- *Policies That Almost Succeed Are Misjudged: Failure Geometry Determines World-Model Policy Evaluation*
- *Evaluating Policies in Imagination: Bias Follows the Near-Miss Share*

**구성(8쪽 + 부록)**
1. 서론
   - 정책 평가의 인프라화, 성공 편향 환각의 선행 보고, 그리고 "어떤 실패인가"라는 질문
   - 그림 1: 실제 성공률이 같은 두 정책이 상상에서 갈리는 예(R2-9의 98.5% 대 58%를 본 실험 수치로 교체)
2. 관련 연구: 2.1절 표의 요약
3. 정의: 실패 거리, 평가 모드, 판정기, 지표(4장)
4. 실패 거리 곡선. 그림 2: 모델 3종 × horizon, 실제 / 상상 AUROC
5. 정책 수준 법칙. 그림 3: o_π 대 n_π, 표 1: 순위 지표, 표준 지표의 맹점
6. 원천과 기전
   - 그림 4: 판정기 × 입력 분해
   - 그림 5: 성공 편중 미세조정의 용량-반응
7. 처방과 checkpoint 선택. 표 2: 노출량을 맞춘 처방, 평가 쪽 할인, regret
8. 한계(시뮬레이션, 결정론적 실패, 공개 모델 수)
- 부록: 과제별 e 정의, 파일럿 사전등록 기록, 판정기 학습, 칸별 결과

**투고 경로**
- 2026-10-21 확인. 2027년 일정은 아직 공식 공지가 없는 것이 많아, 투고 직전에 다시 확인한다.

| 경로 | 투고처 | 일정 | 조건 |
|---|---|---|---|
| 먼저 | arXiv | 12월 중순 | E1·E2(최소 M1 + M2) 완료 |
| 1순위 | RSS 2027(아테네, 7/6–11, 제3자 목록 기준) | 마감 1월 말 예상(2026년은 1/30) | E1~E5 완료 |
| 1순위 대안 | ICML 2027 | 마감 추정 2027-01-22(제3자 추정) | 같음. 평가 방법론 강조 시 |
| 대안 | CoRL 2027(5월 말 예상), NeurIPS 2027 Datasets & Benchmarks | | 두 번째 비디오 월드모델이나 실로봇 사례를 더할 때 |
| 병행 | ICRA 2027 워크숍(서울, 5/24–28; 마감 보통 2~4월) | | 월드모델·평가 워크숍 단편 |

**지원서 서사 연결**
- ECM(VLA 실행 중 실패 탐지와 원인 귀속)은 "실제 실행에서 실패를 잡는" 연구였다. 이 연구는 그 짝으로 "상상 속에서 실패를 놓치는" 문제를 다룬다.
- 두 연구를 합치면 월드모델 기반 평가·감시의 신뢰성이라는 하나의 연구 축이 된다.

---

## 9. 참고문헌 (원문 확인 수준은 NOVELTY_R2 표기와 2026-10-21 추가 확인을 따름)

- Cammarata 2026, *Neural Networks Are Already Clean*, Goodfire. https://already-clean.goodfire.com/
- Xu, Hu, Chen, Gao, Ye, Shi 2026, *World Models Dream of Success: Diagnosing and Repairing Failure Insensitivity in Robot World Models* (CureWM), arXiv 2610.09134. 초록 확인
- Yang et al. 2026, *MiraBench*, arXiv 2605.29360
- Liu et al. 2026, *World-VLA-Loop*, arXiv 2602.06508
- Tu, Wang, Zhu, Jin, Chen 2026, *Beyond Successful Robot Demonstrations: Learning from Human Interactions and Modeling Failure Dynamics* (HF-WM), arXiv 2610.13522. 초록 확인
- Pei et al. 2026, *FARM: Reading Failure Signals from the Internal Predictive States of a Frozen Robotic World Model*, arXiv 2609.11445. 초록 확인
- *Completion Aware Guidance for World Action Models*, NeurIPS 2026(학회 virtual 페이지의 검색 요약 기준)
- Peng et al. 2026 FACT (2608.10232) · Li et al. 2026 DreamTrue (2610.12468) · Li et al. 2026 dWorldEval (2604.22152) · Chen et al. 2026 WorldSync (2608.24885)
- Guo et al. 2025(저자에 Finn), *Ctrl-World*, arXiv 2510.10125(검색 수준)
- *Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning*, arXiv 2601.16163(코드 github.com/nvlabs/cosmos-policy)
- WorldGym (2506.00613) · WorldEval (2505.19017) · RoboWorld (2607.01060) · DreamDojo (2602.06949)
- Li et al. 2024, SIMPLER, CoRL · Navasardyan et al. 2026, FailBench (2609.03611)
- Cao et al. 2019, LDAM, NeurIPS · Chi et al. 2023, Diffusion Policy, RSS · Zhou et al. 2025, DINO-WM
