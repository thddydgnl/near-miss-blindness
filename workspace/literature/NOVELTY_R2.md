# NOVELTY_R2 — 2라운드 세 축(A·B·C)과 예비 축(D·E)의 선행연구·노벨티 검토

- 작성: 2026-10-21 (문헌·노벨티 검토 담당)
- 범위
  - 축 A: Planners live in the near-ties. 모델 기반 계획의 성패는 오차의 크기가 아니라 구조가 정한다
  - 축 B: 월드모델 기반 정책 평가의 near-miss blindness
  - 축 C: Where the action enters decides the race. 행동을 붙이는 위치와 방식이 경주를 정한다
  - 예비 D(시각 월드모델 crushing)와 E(CEM 반복에 따른 순위 붕괴)는 짧게 다룬다
- R1 검토([NOVELTY_R1.md](NOVELTY_R1.md))에서 다룬 문헌은 다시 조사하지 않았다. 필요한 곳에서만 인용한다.
- 파일럿 수치는 각 보고서(R2_2, R2_3·3b, R2_4, R2_5, R2_6, R2_8·8b, R2_9)에서 옮겼다. R2-10, R2-11, R2-12는 진행 중이다.
- 표기
  - **[확인]**: 이번 검토에서 초록이나 본문 페이지를 직접 열었다. 표 번호나 본문 수치를 인용한 곳은 HTML 본문을 열어 확인한 경우다.
  - **[확인, R1]**: R1 검토에서 열어 확인했다. 이번에는 다시 열지 않았다.
  - **[검색]**: 검색 결과의 제목과 요약으로만 보았다. 인용 전에 원문을 다시 확인해야 한다.
- 주의
  - "찾지 못했다"는 이번 검색 범위 안의 결과일 뿐이고, 없다는 증명이 아니다.
  - 2026년 8~10월에 세 축과 겹치는 arXiv가 크게 늘었다. 특히 B축은 10월에만 CureWM, DreamTrue, WorldOracle이 나왔다. 원고를 쓰기 직전에 다시 검색해야 한다.
  - 인용한 2026년 논문 대부분은 동료심사를 거치지 않은 preprint다.

---

## 0. 한눈에 보기

| 축 | 판정 | 가장 위협적인 선행연구 | 한 줄 포지셔닝 |
|---|---|---|---|
| **A** 오차 구조와 near-tie | **중간** (CRN dither와 iteration 축만 보면 중간–높음) | Min, Li, Xu, Du 2026, *When Low Prediction Error Misleads Planning* (arXiv 2610.05550). 끝점 오차를 "후보 풀 중심"과 "행동 상대 응답"으로 나누고, MSE를 지배하는 성분과 결정에 중요한 성분이 다르다고 보였다 | "오차 구조가 중요하다"는 메시지는 이미 붐빈다. 차별점은 항목별 오차 분포를 고정한 채 **후보 간 공유**와 **최적화 iteration 간 지속성**만 바꾸는 2축 대조, 그리고 iteration마다 새로 뽑는 CRN dither라는 값싼 처방이다. 고전 CRN·SAA·stochastic rounding 문헌을 먼저 인정하고, 학습된 월드모델 압축과 표본 기반 MPC로 옮겨 실증한 것으로 기여를 정의한다 |
| **B** near-miss blindness | **중간** (현상 자체는 낮음) | Xu, Hu, Chen, Gao, Ye, Shi 2026, *World Models Dream of Success* (CureWM, arXiv 2610.09134). 공개 체크포인트 4개의 failure insensitivity와 optimism을 보이고 반사실 실패로 고쳤다. 단, 본문에 "policy ranking is untested"라고 적었다 | 성공 편향 환각은 2026년에만 7편 이상이 보고했다. 남은 것은 (i) 실패가 경계에서 얼마나 먼지에 따른 상상 성공 곡선, (ii) 정책의 near-miss 비중이 평가 편향을 정한다는 정책 수준 법칙과 순위 역전, (iii) 판별기의 낙관과 월드모델의 낙관을 분리하는 설계다 |
| **C** 행동 부착 방식 | **중간** | Baldassarre et al. 2025, *Back to the Features: DINO as a Foundation for Video World Models* (DINO-world, arXiv 2507.19468). 행동 없이 사전학습한 latent 예측기에 행동을 붙이는 방식(scratch / 동결 + action block / 전체 FT)을 계획 성공률로 비교했다 | 동결과 전체 FT의 비교 자체는 있다. 없는 것은 같은 ID loss에서의 행동 민감도 비교, 사전학습 쌍둥이와 학습 전 쌍둥이의 대조(선점의 인과 분리), "정체값 = 혁신 몫"이라는 정량 법칙이다. 기전은 modality competition과 gradient starvation의 월드모델판이라고 먼저 밝힌다 |
| D crushing | 중간 (null·자명성 위험) | 직접 선행 없음. 개념상 Elhage et al. 2022 *Toy Models of Superposition*과 원문 에세이 | 월드모델에서 폭 sweep, 혼동(packing) 지표, 뉴런 국소성으로 coarsening과 packing을 가른 연구는 없다 |
| E CEM 순위 붕괴 | **낮음** | Wang et al. 2026, *Decision-Metric Alignment* (2608.18746): PushT에서 CEM 단계별 Spearman 0.403 → 0.227 → 0.036. Zhang et al. 2026 (2610.28011): DINO-WM Wall 성공률이 iteration 11에서 정점을 찍고 30에서 38% 떨어짐 | 단독 논문은 어렵다. 축 A의 iteration 축 증거로 흡수한다 |

---

## 1. 축 A — Planners live in the near-ties: error structure, not error size

### 1.1 주장과 파일럿 근거

**TD-MPC2 + MPPI** (후보 512개, iteration 6회; Meta-World push·pick-place, DMC cheetah·walker)
- R2-3b: 후보 간에 일관된 고정 무작위 unit 제거(D-fix)가 같은 1-step·3-step 오차에서 약한 활성 제거 손상의 100~111%를 재현했다. 약한 활성 고유 효과는 0/8칸이다. 같은 오차의 후보별 독립 제거보다 6/8칸에서 유의하게 더 해로웠다.
- R2-8b(k-bit 활성 양자화, 새 seed, 20 episode)
  - 같은 격자에서 잡음 공유 범위만 다른 확률적 변형 4개는 과제마다 err1이 5% 이내로 같다. 그런데 하락은 mw-push 5~85%p, pick-place 10~75%p, cheetah 47~81%, walker 10~54%로 갈렸다.
  - 후보 공유·forward마다 새로 뽑는 dither(sh-fwd)는 3-bit에서 결정적 반올림 손상의 92%(mw-push)와 82%(pick-place)를 없앴다. DMC(k=4)에서는 10%(cheetah)와 29%(walker)에 그쳤다.
  - 같은 공유 잡음을 한 제어 step의 iteration 6회 동안 고정하면(sh-step) 4/4 과제에서 sh-fwd보다 유의하게 해로웠다. 결정적 반올림과 비교하면 MW에서는 유의한 차이가 없었고, DMC에서는 유의하게 더 나빴다. episode 내내 고정하면(sh-ep) cheetah와 walker에서 결정적 반올림보다 유의하게 해로웠다(+28.3, +40.1%p).
  - 후보 공유 축(CRN)의 이득은 4/4 과제에서 방향이 같았지만, 유의한 과제는 2/4다.

**V-JEPA 2-AC, DROID** (R2-5, 오프라인 단일 평가)
- 1·2위 energy gap의 중앙값은 후보 std의 0.04배다.
- 추론 정밀도만 바꿔도 1위 후보가 바뀐다. fp32 대 fp16은 10%(gap 하위 사분위에서 25%), bf16 대 fp16은 23%다.
- 같은 예측 변화량에서 순위 손상은 "어떤 unit을 건드렸나"가 아니라 후보 간 차등 성분이 정했다. 약한 활성 대 후보 일관 무작위의 비는 0.85~1.06이고, 독립 mask는 1.5~2.6배 더 해로웠다.
- **한계:** V-JEPA 쪽은 반복 CEM을 돌리지 않았다. iteration 축의 근거는 아직 TD-MPC2에만 있다.

### 1.2 가장 가까운 선행연구

| # | 논문 | URL | 이미 보인 것 | 확인 |
|---|---|---|---|---|
| 1 | Min, Li, Xu, Du, *When Low Prediction Error Misleads Planning: Diagnosing Representation, Dynamics, and Decision Failures in Latent World Models*, arXiv 2610.05550 (2026-10-04) | https://arxiv.org/abs/2610.05550 | 같은 물리 출발점에서 후보 300개를 공유하고, 끝점 오차를 후보 풀 중심 오차와 행동 상대 응답으로 나눴다. 중심 오차가 16칸 중 14칸에서 MSE를 지배한다. 그중 6칸에서는 행동 상대 응답만 고치는 oracle이 중심만 고치는 oracle보다 순위 상관과 top-30 elite 품질을 더 올린다. LeWM 실행 실험에서는 반대로 중심 수리가 나은 설정도 있었다. 결론은 "오차 크기와 결정 효과는 별개이므로 표현, 예측, 계획을 따로 평가하라"이다. 초록에는 양자화, iteration 간 지속성, CRN, near-tie가 없다 | [확인] |
| 2 | Wang, Fan, Huang, *Not All Errors Matter: Decision-Relevant Prediction Error Predicts Planning Quality*, arXiv 2609.32322 (2026-09) | https://arxiv.org/abs/2609.32322 | iso-error 프로토콜로 총 오차를 고정하고 차원별 배분만 바꿨다. 총 오차와 계획 성공의 ρ = −0.25, 결정 관련 오차(DRPE)는 ρ = −0.84다. 총 오차 1% 차이가 성공률 97% 대 37%로 갈린다. 깊은 상상 rollout이 결정 관련 오차를 증폭한다. factored gridworld, 이름 없는 표준 planner | [확인] |
| 3 | You et al., *A Control Theory of Predictability in Latent World Models*, arXiv 2607.10362 (2026-07) | https://arxiv.org/abs/2607.10362 | 데이터 평균 예측 오차는 planner의 suboptimality를 bound하지도 추적하지도 못한다. committed plan에서의 예측·실제 plan cost 차이가 suboptimality를 2배로 bound한다. latent MPC에서 1-step 검증 오차는 제어 성공과 거의 무상관이다 | [확인] |
| 4 | Fu et al., *An Empirical Study of World Model Quantization*, arXiv 2602.02110; Xu, Guo, Chen, Du, Shi, *Predict Before You Deploy* (PreDE), arXiv 2609.19441 | https://arxiv.org/abs/2602.02110 ; https://arxiv.org/abs/2609.19441 | Fu et al.: DINO-WM에 PTQ를 하면 공격적 비트에서 계획 목표와 과제 성공이 어긋난다. PreDE: world action model 5개, PTQ 설정 28개(보류)에서 오프라인 행동 편차로 closed-loop 저하를 accept/reject/defer한다. 결정한 21개(75%)는 모두 맞았다. "비트 수나 단일 편차 문턱으로는 저하를 설명할 수 없다"고 결론짓는다. 두 연구 모두 반올림 방식(결정적·확률적)과 오차의 공유·지속 구조는 다루지 않는다 | Fu [확인, R1], PreDE [확인] |
| 5 | 고전 CRN과 정책 탐색: Strens & Moore, *Policy Search using Paired Comparisons*, JMLR 2002; Ng & Jordan, *PEGASUS*, UAI 2000; Kleinman, Spall, Naiman, *Simulation-Based Optimization with Stochastic Approximation Using Common Random Numbers*, Management Science 1999; Nelson & Matejcik, Management Science 1995; Smith & Winkler, *The Optimizer's Curse*, Management Science 2006 | https://www.jmlr.org/papers/v3/strens02a.html ; https://arxiv.org/pdf/1301.3878 ; https://pubsonline.informs.org/doi/fpi/10.1287/mnsc.45.11.1570 ; https://pubsonline.informs.org/doi/fpi/10.1287/mnsc.41.12.1935 ; https://ideas.repec.org/a/inm/ormnsc/v52y2006i3p311-322.html | PEGASUS는 시작 상태와 난수열을 고정해 확률적 정책 최적화를 결정적 문제로 바꾼다. Strens & Moore는 그 "overfitting"을 지적하고 paired comparison으로 개선한다. CRN은 SPSA·FDSA 수렴을 빠르게 하고(Kleinman et al.), 최선 시스템 선택(ranking & selection)의 표본 수를 줄인다(Nelson & Matejcik). 추정치를 최적화해 고르면 고른 대안의 값이 과대평가된다(Smith & Winkler). **축 A의 처방(후보 간 공유 + iteration마다 새로)은 "비교 안에서는 난수를 공유하라"와 "고정된 scenario에 최적화하지 마라"라는 두 고전 원리의 결합이다** | Strens & Moore [확인], 나머지 [검색] |
| 6 | 반올림 잡음과 반복 알고리즘: Gupta et al., *Deep Learning with Limited Numerical Precision*, ICML 2015; *The Positive Effects of Stochastic Rounding in Numerical Algorithms* (HAL); 2026년 SR 오차 분석(arXiv 2603.06060, 제목 미확인) | https://arxiv.org/abs/1502.02551 ; https://hal.archives-ouvertes.fr/hal-03716058 ; https://arxiv.org/pdf/2603.06060 | 16-bit 고정소수점 학습은 stochastic rounding이 있어야 된다(Gupta et al.). Euler 전진법에서 결정적 반올림과 편향된 SR은 iteration에 걸쳐 오차가 쌓이지만, 편향 없는 SR은 쌓이지 않는다(HAL). 합산 오차는 round-to-nearest에서 n, SR에서 √n으로 자란다(2026 분석). **"지속 편향은 반복에서 쌓이고 새로 뽑는 잡음은 상쇄된다"는 수치해석의 표준 결과다** | [검색] |

**함께 볼 연구**
- 모델 착취(planner가 모델 오차를 이용함)
  - Boney et al., *Regularizing Trajectory Optimization with Denoising Autoencoders*, NeurIPS 2019 [확인: https://arxiv.org/abs/1903.11981]. 학습된 모델로 궤적을 최적화하면 모델의 부정확성을 이용한다. 검색 결과의 PDF 요약에는 "더 효율적인 최적화기일수록 이 함정에 잘 빠진다"는 서술도 있었다 [검색].
  - Arumugam et al., *Mitigating Planner Overfitting in Model-Based RL*, 2018 [검색: https://arxiv.org/pdf/1812.01129]
  - Ha & Schmidhuber, *World Models*, 2018 [검색: https://arxiv.org/abs/1803.10122v1]. VizDoom에서 controller가 꿈속 모델의 결함을 이용하자, MDN-RNN의 temperature를 올려(꿈에 잡음을 넣어) 착취를 막았다. "잡음 주입이 착취를 줄인다"의 초기 사례다.
  - Obst & Stolzenburg, *In CEM, a World Model Is Also a Proposal Mechanism*, arXiv 2610.00921 [확인: https://arxiv.org/abs/2610.00921]. CEM에서 scoring 오류는 현재 결정과 다음 iteration의 후보 분포를 모두 바꾼다. Walker·Cheetah에서 iteration이 진행돼도 elite 일치가 좋아지지 않았다.
  - Li et al. 2026, *Action from Adjacent Set...* [확인, R1: https://arxiv.org/abs/2607.23602]. 후보 풀을 키우면 모델 오류가 이길 기회가 늘어난다(후보 공간의 optimizer's curse).
  - Duan, Zhang, Luo, *TRAP*, arXiv 2605.01950 [확인: https://arxiv.org/abs/2605.01950]. 소수의 결정적 궤적 순위만 흔들어도 DreamerV3·TD-MPC2의 계획을 탈취한다(백도어 공격). 순위 구조가 long-tail이라는 주장이다.
- 정밀도와 near-tie
  - Yuan et al., *Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference* (v1 제목 *Give Me FP32 or Give Me Death?*), arXiv 2506.09501 [확인: https://arxiv.org/abs/2506.09501]. bf16 greedy decoding에서 GPU 수·종류·batch 크기만 바꿔도 정확도가 최대 9% 변한다. 해법으로 계산만 FP32로 하는 LayerCast를 제안한다. 분기 지점에서 top-1과 top-2 확률이 정밀도 오차로 뒤바뀐다는 그림 설명은 검색 요약에서 보았다 [검색]. NeurIPS 2025 oral 목록 [검색: https://nips.cc/virtual/2025/oral/118170].
  - Dutta et al., *Accuracy is Not All You Need*, NeurIPS 2024 [검색: https://arxiv.org/html/2407.09141v1]. 압축 모델은 정확도가 같아도 답이 대량으로 뒤집힌다("flips"). 압축 평가에 KL과 % flips를 쓰자고 제안한다.
  - Gao, Yang, Kingston, *Sampling-based Kinodynamic Planning with World Models*, arXiv 2610.12910 [확인: https://arxiv.org/html/2610.12910v1]. DINO-WM PushT CEM에서 성공 수는 fp32 45/50, fp16 46/50, tf32 44/50이고 fp16이 3.4배 빠르다. 집계 성공률은 정밀도에 둔감하다. R2-5의 "정밀도가 1위를 바꾸지만 regret은 작다"와 같은 그림이다.
- 실제 로봇 월드모델의 계획 지형
  - V-JEPA 2 원논문, arXiv 2506.09985 [확인: https://arxiv.org/html/2506.09985]. energy 지형이 "relatively smooth and locally convex"라고 서술한다. CEM은 표본 800개, iteration 10회, horizon 1이다. 추론 정밀도는 언급하지 않는다.
  - ARC-Bench, arXiv 2609.05461 [확인: https://arxiv.org/abs/2609.05461]. frozen JEPA-WM 공식 체크포인트에서 1위 후보가 대개 최적이 아니고, 잦은 재계획이 이 결함을 가린다.
- 정책(VLA) 양자화: VLAQuantBench (arXiv 2609.25376) [검색], QVLA (arXiv 2602.03782, ICLR 2026) [검색], DA-PTQ (arXiv 2604.11572) [검색]. closed-loop에서의 시간적 오차 누적을 지적하지만, 처방은 비트 배분과 보정이다. 확률적 반올림 처방은 없다.
- 혼동 주의(행동 표본 쪽 잡음 구조): deterministic samples를 쓰는 sampling-based MPC (arXiv 2601.03893) [검색], *Variance-Reduced MPPI* (arXiv 2602.03639, RSS 2026) [검색]. 이들은 행동 표본의 분산을 다룬다. 모델 오차의 공유·지속 구조와는 다른 문제다. 리뷰어가 섞어 읽지 않도록 본문에서 구분해야 한다.

### 1.3 이미 보인 것과 남은 공백

**이미 보인 것**
- 예측 오차의 크기는 계획 품질을 잘 대변하지 못한다. 2026년 7~10월에만 You et al., DRPE, Min et al.이 나왔고, R1에서 본 AD-WM, Prediction Fidelity(NeurIPS WS), ARC-Bench, Decision-Metric Alignment도 같은 메시지다.
- 후보 공통 성분과 행동 상대 성분의 분해(Min et al.). R2-5의 "공통/차등 성분" 분해와 같은 축이다.
- 비교 안에서는 CRN이 분산을 줄이고, 고정된 scenario에 최적화하면 그 잡음에 과적합한다(고전 OR과 정책 탐색).
- 결정적 반올림의 지속 편향은 반복 알고리즘에서 누적된다(수치해석).
- 월드모델 양자화는 계획을 망가뜨리고, 그 정도는 비트 수만으로 설명되지 않는다(Fu et al., Ranganath et al., PreDE).
- 정밀도 차이가 argmax near-tie를 뒤집는다(LLM 연구).

**남은 공백**
1. 항목별 오차 분포를 고정한 채 (a) 후보 간 공유와 (b) 최적화 iteration 간 지속성만 바꾸는 2축 대조를 학습된 월드모델 계획에서 한 연구는 찾지 못했다. Min et al.은 (a)만 oracle 수리로 다룬다.
2. "공유 잡음은 iteration마다 새로 뽑아야 한다"(sh-fwd 대 sh-step)는 결과는 찾지 못했다. CRN-SPSA는 개념상 같은 구조지만, 학습된 모델의 압축 오차나 표본 기반 MPC에 적용한 보고는 없다.
3. 저비트 월드모델(또는 VLA) 추론의 처방으로 확률적 반올림이나 dither를 쓴 연구는 없다.
4. 실제 로봇 월드모델에서 결정 near-tie의 밀도와 정밀도 flip 비율을 잰 보고는 없다. 가장 가까운 것은 LLM 연구와 DINO-WM 집계 성공률(2610.12910)이다.

### 1.4 판정: **중간** (iteration 축과 CRN dither만 보면 중간–높음)
- 상위 메시지("오차 크기가 아니라 구조")는 2026년 하반기에 이미 붐빈다. 후보 축 분해는 Min et al.이 먼저 냈다.
- 처방의 원리(CRN + 새 표본)는 시뮬레이션 최적화와 수치해석에서 고전이다. 원리 자체를 새롭다고 주장하면 리뷰에서 바로 반박된다.
- 그래도 다음은 새롭다.
  - 같은 err1에서 공유 범위만 바꿔 성공률 하락이 5%p에서 85%p까지 갈린다는 통제 실험
  - iteration 단위의 새로움이 필요하다는 결과(sh-step이 결정적 반올림만큼 나쁘다)
  - 한 줄짜리 처방으로 3-bit 조작 과제 손상의 82~92%를 없앤 효과 크기
- 이 결과가 DMC 1~2 과제, 체크포인트 1개, iteration dose-response 없음에 머물면 "중간"이다. 아래 1.6의 1~3번을 채우면 "중간–높음"으로 올라간다.

### 1.5 가장 위협적인 논문
**Min et al. 2026 (arXiv 2610.05550).**
- 같은 커뮤니티(latent 월드모델 계획)에서 같은 분해(공통 대 행동 상대)와 같은 결론(크기 ≠ 결정 효과)을 2주 전에 냈다.
- 리뷰어는 "후보 축은 Min et al., iteration 축은 교과서적 SA/CRN"이라고 쓸 수 있다.
- 개념상 가장 위협적인 고전은 Strens & Moore 2002(고정 scenario 과적합과 paired comparison)와 Kleinman et al. 1999(CRN-SA)다. OR 배경의 리뷰어가 인용할 가능성이 높다.

### 1.6 포지셔닝

**제목 후보**
- *Share Across Candidates, Refresh Across Iterations: Error Structure Decides Planning with Low-Bit World Models*
- *Fresh, Shared, or Persistent? Common-Random-Number Dithering for World-Model MPC*

**주장(한 문장)**
같은 k-step 예측 오차에서, 표본 기반 계획기의 손상은 월드모델 오차가 후보 간에 공유되는지와 최적화 iteration 간에 지속되는지로 정해진다. 후보 간에 공유하고 iteration마다 새로 뽑는 CRN dither는 조작 과제에서 3-bit 활성 양자화 손상의 대부분을 없앤다.

**서술 원칙**
- 서론 첫 단락에서 고전 원리(CRN, SAA의 optimizer's curse, stochastic rounding)를 인정한다. 기여는 "학습된 월드모델 압축 오차와 MPPI/CEM으로 옮겨, 두 축을 분리하고 처방을 실증한 것"으로 정의한다.
- Min et al.과의 관계를 명시한다. 그들은 후보 축을 oracle 수리로 보았다. 이 연구는 iteration 축을 더하고, 실제 오차원(압축, 정밀도)과 처방을 준다.
- 원문 에세이와의 연결은 한 단락으로 제한한다.
  - "약한 활성은 특별하지 않다"는 계획에서도 성립한다(R2-3b 0/8, R2-5 비 0.85~1.06).
  - "near-tie는 중요하지 않다"는 전제는 계획에서 성립하지 않는다. V-JEPA 2-AC의 1위 결정은 대부분 near-tie이고, 반복 계획기는 near-tie에서 지속 편향을 따라간다.

**안전하려면 필요한 실험(우선순위 순)**
1. **iteration 수 dose-response.** MPPI iteration을 {1, 2, 3, 6, 12}로 바꿔 sh-step − sh-fwd 차를 잰다. iteration 1에서는 0이고 iteration과 함께 커져야 한다. 공유 표를 1/2/3/6 iteration마다 갈아 끼우는 refresh 주기 sweep도 둔다. 기전 주장의 핵심 검정이다(R2-8b 다음 단계 2번).
2. **이론 부록.** 2차 목적함수에 (i) 상수 편향(지속 오차), (ii) iteration마다 새 잡음(SA형), (iii) 후보별 독립 잡음을 넣고, CEM/MPPI 평균 이동의 기댓값과 분산을 계산한다. Smith–Winkler의 optimizer's curse와 SAA 대 SA의 틀에 맞춘다.
3. **두 번째 계획기와 월드모델.**
   - DINO-WM + CEM: Fu et al.의 설정(W4A4, 활성 3-bit)에 같은 공유 범위 변형을 넣는다.
   - V-JEPA 2-AC 공식 CEM(표본 800, iteration 10): bf16과 fp32의 near-tie flip이 반복 CEM 뒤 최종 행동과 regret을 바꾸는지 잰다. 정밀도 잡음 자체가 후보 공유형인지, iteration마다 새로운지도 측정한다.
4. **강한 기준선.** per-channel 양자화, 보정형 PTQ(AWQ/OmniQuant류), QAT, 그리고 "결정적 반올림 + iteration마다 바뀌는 공유 격자 offset". CRN dither가 이들과 겹치는지, 보완하는지 보여야 한다.
5. **지표 검정.** k-step 오차, CVB, Min et al.식 중심/상대 분해가 공유 범위 변형을 구별하지 못함을 보인다. "후보 차등 오차의 iteration 간 자기상관" 같은 구조 지표가 손상을 예측하는지는 사전등록해 검정한다.
6. **규모.** MW 6개 이상, DMC 2개 이상, 체크포인트 seed 3개, 30 episode. DMC는 k=5를 사전등록 칸으로 둔다(R2-8b 권고).
7. **실측 비용.** 정수 커널에서 공유 dither의 지연을 잰다. 층·forward마다 난수 벡터 하나뿐이고 후보 수와 무관하다는 점이 실용성의 근거다.

**피해야 할 주장**
- "k-step 오차는 틀린 지표다." R2-8의 156조건 전체에서 err1의 Spearman은 0.70으로 쓸 만했다. "같은 오차 수준 안에서는 구조가 정한다"만 주장한다.
- "dither는 일반적으로 좋다." 독립 dither는 walker에서 결정적 반올림보다 나빴고(+28.4%p), sh-step도 결정적 반올림보다 나쁠 수 있다.
- "새로운 원리." CRN, SAA, stochastic rounding은 고전이다. 새로운 것은 대상(학습된 월드모델의 압축 오차), 축의 분리, 효과 크기다.

### 1.7 피지컬 AI 랩 어필
- **학습된 월드모델로 MPC를 하는 그룹**: TD-MPC2 계열, DINO-WM·JEPA-WM 계열(V-JEPA 2-AC를 만든 Meta, 그리고 *What Drives Success in Physical Planning with Joint-Embedding Predictive World Models?* [확인: https://arxiv.org/abs/2512.24497; 저자 Terver, Yang, Ponce, Bardes, LeCun]), LeWM 기반 decision-alignment 연구자들. 계획 실패의 원인을 진단하는 새 대조 설계를 준다.
- **on-robot 저비트 추론과 압축**: Fu et al.(검색 결과의 저자 이메일이 Huawei 도메인), PreDE와 CureWM 저자 그룹, VLA 양자화 벤치마크 그룹. "같은 비트 수에서 반올림 방식만 바꿔 pick-place 성공률 45% → 90%"는 바로 쓸 수 있는 손잡이다.
- **표본 기반 제어(MPPI/CEM)와 OR 배경 그룹**: SA 대 SAA, CRN의 관점으로 쓴 분석은 제어·최적화 리뷰어에게 설득력이 있다.
- **해석가능성 커뮤니티**: 원문 에세이의 near-tie 전제를 실제 로봇 모델에서 반증하고, "약한 활성은 특별하지 않다"는 쪽은 지지한 결과다.

---

## 2. 축 B — Near-miss blindness in world-model policy evaluation

### 2.1 주장과 파일럿 근거
- R2-4·R2-6(2D 집어 들기, 결정론적 실패, 실패율 p = 0.01)
  - 경계 바로 바깥(정렬 오차가 반폭의 1~1.2배)의 near-miss 실패를 성공으로 그리는 비율: 상태 입력 99.6%, 고정 DINOv2 특징 92.5%
  - far-miss는 5.4%, 6.0%
  - 결정 경계 r*: 1.02 → 1.39(상태), 1.04 → 1.27(시각)
- R2-9(DINO-WM식 latent 비디오 예측기, 고정 성공 판별기, p = 0.03)
  - 정책 6개 평균 과대 추정 +16.9%p
  - 실제 성공률 52.6%와 48.2%인 두 정책이 상상에서는 98.5%와 58.0%로 평가됐다.
  - near-miss 비중과 과대 추정의 상관 r = 0.95
  - 판별기는 실제 프레임에서 정확도 1.000이다. 과대 추정은 모두 상상에서 생긴다.
  - 학습 4배로 −52%, 재가중 −64%, 대비 쌍 −53%. 처방 효과는 구조에 따라 뒤집혔다. R2-6 시각 MLP에서는 재가중이 편향을 키웠다(0.48 → 0.71).
- R2-12(DINO-WM PushT 공개 체크포인트)가 진행 중이다.

### 2.2 가장 가까운 선행연구

| # | 논문 | URL | 이미 보인 것 | 확인 |
|---|---|---|---|---|
| 1 | Xu, Hu, Chen, Gao, Ye, Shi, *World Models Dream of Success: Diagnosing and Repairing Failure Insensitivity in Robot World Models* (CureWM), arXiv 2610.09134 (2026-10-06) | https://arxiv.org/abs/2610.09134 ; https://arxiv.org/html/2610.09134 | 두 구조 계열의 공개 체크포인트 4개(Cosmos Policy, Ctrl-World)가 실행으로 검증한 실패에 성공 같은 예측을 낸다. 성공 시연에서 severity grid로 대안 행동을 만들고 실행으로 결과를 검증한 뒤 post-hoc fine-tune한다. LIBERO 실패 반사실 484개에서 optimism이 공식 데이터 미세조정 기준선 80%에서 CureWM 평균 38%(30~43%)로 줄었다. 실제 로봇에서는 성공 시연만으로 미세조정한 기준선 90%가 33%로 줄었다(latent 거리 진단 기준). **본문:** severity 0.4~1.0별 AUROC를 보면 이득이 높은 severity에 몰린다. 경계에 가장 가까운 severity 0.2는 실패가 2개 이하라 분석에서 뺐다. 6.4절에 "policy ranking is untested", best-of-N 선택의 이득은 통계적으로 확인되지 않았다고 적었다 | [확인] |
| 2 | Yang et al., *MiraBench: Evaluating Action-Conditioned Reliability in Robotic World Models*, arXiv 2605.29360 (2026-05) | https://arxiv.org/abs/2605.29360 ; https://arxiv.org/html/2605.29360v1 | 세 수준(physics adherence, action-following, optimism bias detection), 12개 구성, 인간 판정 16,000개 이상. optimism bias가 만연하다. 실패 유도 행동은 성공 행동에 6종 교란(grip 부족, 조기 release, carry slip 등)을 **severity s = 0.5 하나로** 준 것이다. near/far 비교와 정책 순위 분석은 없다. 원인 가설은 "decoupling hypothesis"다. 장면 조건의 성공 prior가 행동 조건 잔차를 압도한다. 성공 데이터로 post-training하면 failure preservation이 떨어진다(DreamDojo-2B 48.7 → 23.1) | [확인] |
| 3 | Liu, Bai, Ci, Ma, Shou, *World-VLA-Loop: Closed-Loop Learning of Video World Model and VLA Policy*, arXiv 2602.06508 (2026-02) | https://arxiv.org/abs/2602.06508 ; https://arxiv.org/html/2602.06508v1 | 비디오 월드모델이 "minor action errors"에서 오는 실패를 시뮬레이션하지 못하고, 잘못된 행동에도 성공을 환각한다(Cosmos-Predict 2). **near-success 실패를 명시적으로 문제로 삼는다.** 성공과 near-success 궤적을 섞은 SANS 데이터와 reward head로 고친다. LIBERO-Object에서 near-success 데이터를 빼면 정렬이 85/95%에서 60/65%로 떨어진다(표본이 작다). 실패 거리에 따른 정량 곡선과 정책 평가 편향은 다루지 않는다. 목적은 VLA RL용 환경이다 | [확인] |
| 4 | 성공 편향 환각과 실패 데이터 처방: Peng, Liang, Yan, Hansen, Wang, *FACT*, arXiv 2608.10232; Li et al., *DreamTrue*, arXiv 2610.12468; Li, Zhou, Chen, Xue, Zhu, *dWorldEval*, arXiv 2604.22152 | https://arxiv.org/abs/2608.10232 ; https://arxiv.org/abs/2610.12468 ; https://arxiv.org/html/2604.22152v1 | FACT: 실패 롤아웃으로 행동 결과를 감독해 "success-biased future hallucination under bad actions"를 줄인다. DreamTrue: 실패 상호작용의 희소한 coverage가 예측을 성공 쪽으로 기울인다. 보정, 반사실 post-training, reward model RL을 합친 방법으로 인간 평가 상호작용 결함률을 48.12% → 6.25%로 낮췄다. dWorldEval: 기존 월드모델이 "ignore erroneous actions and hallucinate successful outcomes"한다고 명시한다. 실패 1k개를 더하고 progress token으로 성공을 판정해 Pearson 0.91~0.93, MMRV 0.013을 보고한다 | [확인] |
| 5 | 정책 평가용 월드모델: WorldGym (arXiv 2506.00613), WorldEval (2505.19017), Ctrl-World (2510.10125), Veo world simulator (2512.10675), RoboWorld (2607.01060), GigaWorld-1 (2607.02642), DreamDojo (2602.06949) | https://arxiv.org/abs/2506.00613 ; https://arxiv.org/abs/2505.19017v1 ; https://arxiv.org/pdf/2510.10125 ; https://arxiv.org/abs/2512.10675 ; https://arxiv.org/abs/2607.01060 ; https://huggingface.co/papers/2607.02642.md ; https://arxiv.org/abs/2602.06949 | 모두 실제 성공률과의 상관이나 순위 보존을 보고한다. 예: RoboWorld Pearson 0.989, Spearman 0.970. GigaWorld-1은 평가기 품질을 "long-horizon, action-faithful rollout consistency"가 지배한다고 정리한다. near-miss 구조로 층화한 분석은 없다. Veo 시뮬레이터는 절대 성공률을 대체로 **과소** 추정하고 순위는 유지했다(Pearson 0.88)는 요약이 검색되었다. 낙관 편향이 보편적이지 않을 수 있다는 반례 후보다 | WorldGym·Veo·RoboWorld·GigaWorld·DreamDojo [확인], WorldEval·Ctrl-World [검색], Veo의 수치 [검색] |
| 6 | 판별기 쪽 낙관과 평가 지표: Navasardyan, Danielyan, Davtyan, *FailBench*, arXiv 2609.03611; Li et al., *SIMPLER*, CoRL 2024 | https://arxiv.org/abs/2609.03611 ; https://arxiv.org/abs/2405.05941 | FailBench: VLM failure detector 13개 중 최고도 balanced accuracy 0.77이고, 증거가 모호하면 성공으로 예측하는 편향이 있다. 접촉이 많은 조립 과제에서는 0.60 미만이다. SIMPLER: MMRV는 "실제 성공률이 비슷한 정책 사이의 순위 위반"에 관대하도록 설계됐다. **B의 핵심 사례(실제 52.6% 대 48.2%가 상상 98.5% 대 58.0%)는 MMRV로는 거의 벌점을 받지 않는다** | FailBench [확인], SIMPLER [검색] |

**함께 볼 연구**
- Chen et al., *Do Robotic World Models Really Follow Actions?* (WorldEcho/WorldSync), arXiv 2608.24885 [확인: https://arxiv.org/html/2608.24885]. "overly optimistic futures that deviate from the conditioned actions"를 실패 모드로 적는다.
- WMPO (arXiv 2511.09515, ICLR 2026) [검색], HaWMPO (arXiv 2609.09941) [확인: https://arxiv.org/abs/2609.09941]. 월드모델 안에서 VLA를 RL로 고친다. 상상 롤아웃의 환각이 정책 학습을 오도한다는 문제의식이다. near-miss 낙관은 reward hacking 경로이기도 하다.
- WorldOracle, arXiv 2610.18826 [확인: https://arxiv.org/abs/2610.18826]. 같은 초기 관측과 저수준 행동열로 실제 실행(성공과 실패 포함)과 비교하는 ACWM 벤치마크다. 4차원 평가, 7개 모델. 정책 평가는 아니다.
- Foresight, arXiv 2606.23085 [확인]. 행동 조건 월드모델 latent로 실패를 감지한다.
- No Free Checker (검증기 survey), arXiv 2609.09250 [확인]. 검증기의 availability와 credibility가 trade-off라고 정리한다.
- 클래스 불균형 이론: Cao et al., *LDAM*, NeurIPS 2019 [검색: https://arxiv.org/abs/1906.07413v2]. 불균형에서 소수 클래스 쪽 margin을 키워 결정 경계를 옮긴다. B의 경계 팽창(r* 이동)은 이 고전 현상의 생성 월드모델판이다. 논문에서 이 연결을 먼저 밝히면 리뷰가 쉬워진다.

### 2.3 이미 보인 것과 남은 공백

**이미 보인 것**
- 성공 위주 데이터로 학습한 월드모델은 실패 행동에도 성공을 예측한다. MiraBench, CureWM, FACT, DreamTrue, World-VLA-Loop, dWorldEval, WorldSync가 2026년 2~10월에 보고했다.
- 실패, 반사실, near-success 데이터로 이를 고친다(CureWM, FACT, DreamTrue, World-VLA-Loop, dWorldEval).
- 월드모델 기반 정책 평가의 순위 상관은 대체로 높게 보고된다(0.88~0.995).
- VLM 성공 판별기도 성공 쪽으로 편향된다(FailBench).

**남은 공백**
1. **실패 거리 곡선.** 경계로부터의 거리에 따른 P(상상 성공 | 실제 실패)를 촘촘히 잰 연구는 없다. MiraBench는 severity가 하나뿐이다. CureWM은 경계 근처를 표본 부족으로 뺐다. World-VLA-Loop는 정성적이다.
2. **정책 수준의 결과.** 정책의 near-miss 비중이 평가 편향을 정한다는 법칙(r = 0.95), 실제 성공이 같은 정책이 상상에서 뒤집히는 순위 역전을 보인 연구는 없다. CureWM은 policy ranking을 미검증으로 남겼다.
3. **편향 원천의 분리.** 실제 프레임에서 완벽한 판별기로 판별기 낙관을 제거하고 월드모델 낙관만 잰 설계는 없다.
4. 학습 길이, 노출량을 맞춘 처방 비교, 구조에 따라 처방 효과가 뒤집히는 현상(MLP와 ViT 예측기에서 재가중의 효과가 반대)은 보고되지 않았다.
5. 공개 latent 월드모델(DINO-WM)에서의 검증은 없다.

### 2.4 판정: **중간** (현상 자체만 보면 낮음)
- "월드모델은 실패를 성공으로 그린다"는 더는 새롭지 않다. 이것을 기여로 내세우면 낮음이다.
- 실패 거리 곡선과 정책 수준 법칙은 새롭다. 공개 모델과 실제 학습 정책 checkpoint로 보이면 중간–높음까지 갈 수 있다.
- 위험 두 가지
  - **시간.** CureWM 저자들이 정책 순위를 다음 과제로 남겼다. 같은 저자군이 PreDE도 냈다(Xu, Chen, Shi 공통). GigaWorld나 RoboWorld 그룹도 near-miss 층화를 빠르게 더할 수 있다.
  - **규모.** 정책 수준 결과(R2-9)는 2D 장면에서 직접 만든 정책 6개다. R2-12가 핵심 관문이다.

### 2.5 가장 위협적인 논문
**CureWM (Xu et al. 2026, arXiv 2610.09134).**
- 같은 현상, 같은 이름의 지표(optimism), severity grid, 정책 평가에 쓰이는 공개 체크포인트(Ctrl-World), 실제 로봇 검증, 수리 방법까지 갖췄다. 2주 전에 공개됐다.
- 다만 경계 근처를 다루지 못했고 정책 순위를 재지 않았다. 이 두 빈칸이 B축이 설 자리다.

### 2.6 포지셔닝

**제목 후보**
- *Near-Miss Blindness: World-Model Policy Evaluation Overrates Policies That Almost Succeed*
- *Evaluating Policies in Imagination: Bias Follows the Near-Miss Share*

**주장(한 문장)**
월드모델이 실패를 성공으로 상상할 확률은 그 실패가 결정 경계에서 얼마나 먼지의 함수다. 따라서 정책 평가 편향은 정책의 성공률이 아니라 실패의 기하(near-miss 비중)가 정하고, 실제 성공률이 같은 정책도 상상에서는 순위가 임의로 갈린다. 상관과 MMRV 같은 표준 지표는 이것을 잘 보지 못한다.

**서술 원칙**
- 첫 단락에서 성공 편향 환각을 MiraBench, CureWM, FACT, World-VLA-Loop, dWorldEval이 보였다고 인정한다.
- 기여는 "얼마나 먼 실패까지 성공으로 보이는가"의 곡선과, 그것이 정책 평가와 checkpoint 선택에 주는 결과로 정의한다.
- LDAM류 클래스 불균형 이론과의 연결(경계가 소수 클래스 쪽으로 밀림)을 명시한다.

**안전하려면 필요한 실험**
1. **공개 모델 2종 이상.** DINO-WM PushT(R2-12) + 정책 평가에 실제로 쓰이는 비디오 월드모델 1종. CureWM과 같은 Ctrl-World LIBERO 체크포인트나 Cosmos Policy를 쓰면 직접 비교가 된다. 시뮬레이터로 실패 거리를 계산할 수 있는 LIBERO류가 좋다.
2. **실제 학습 궤적의 정책 checkpoint.** 손으로 설계한 정책 대신 학습 초·중·후 checkpoint를 쓴다. 정책이 좋아질수록 near-miss 비중이 늘고 과대 추정이 커지는지 본다. "정책 개선 루프에서 가장 과대평가되는 단계"라는 주장은 이 실험으로만 선다.
3. **경계 근처를 포함한 실패 거리 곡선.** CureWM이 뺀 구간까지 severity를 촘촘히 두고, 시뮬레이터 반사실로 near-miss 표본을 충분히 만든다. P(상상 성공 | 거리)와 AUROC를 보고한다.
4. **판별기 분해(2×2).** (a) 실제 프레임 완벽 판별기, (b) VLM 판별기 on 실제 프레임, (c) VLM 판별기 on 상상 프레임. 편향을 월드모델 몫과 판별기 몫으로 나누고 FailBench 결과와 연결한다.
5. **지표 제안.** near-miss로 층화한 보정 오차, 실패 거리 가중 MMRV. 표준 MMRV가 이 편향을 놓치는 사례를 수치로 보인다.
6. **노출량을 맞춘 처방 비교.** 재가중, near-miss 반사실(CureWM식), 대비 쌍, 평가 쪽 보정(정책 행동의 경계 근접도로 성공 추정을 할인). 구조(MLP, ViT, 비디오 확산)별로 처방 순위가 바뀌는지 본다.
7. **확률적 실패.** 마찰 잡음 등으로 결정론적 실패 가정을 없앤다.

**피해야 할 주장**
- "월드모델은 낙관적이다"(일반 명제): Veo 시뮬레이터는 과소 추정을 보고했다는 요약이 있다. "실패 기하에 따른 조건부 편향"으로 쓴다.
- "최초 발견": 위 7편이 있다.
- "통합 지연(consolidation delay)이 기전이다": R2-9의 ViT 예측기에서 지지되지 않았다(ρ = −0.43).

### 2.7 피지컬 AI 랩 어필
- **월드모델로 정책을 평가하고 고르는 그룹**: Ctrl-World(Stanford·Tsinghua, 저자에 Finn; 검색 결과 기준), WorldGym(저자에 Percy Liang, Sherry Yang), Gemini Robotics 팀(Veo 평가), DreamDojo(저자에 Linxi Fan), RoboWorld(저자에 Kimin Lee), WorldEval·dWorldEval(저자에 Yichen Zhu), GigaWorld, AgiBot(GE-Sim).
- **월드모델 안에서 RL로 VLA를 고치는 그룹**: WMPO, World-VLA-Loop, HaWMPO. near-miss 낙관은 정책이 "거의 성공"으로 수렴하게 만드는 reward hacking 경로다.
- **안전·검증 연구자**: 검증기의 신뢰도를 실패 기하로 층화해 보라는 메시지다.
- 실무 메시지: 월드모델 점수로 checkpoint를 고르면 near-miss로 실패하는 checkpoint를 고르게 된다.

---

## 3. 축 C — Where the action enters decides the race

### 3.1 주장과 파일럿 근거 (R2-2; R2-10 진행 중)
- 설정: 2D 밀기, 고정 DINOv2 토큰, 6층 ViT 예측기, zero-init AdaLN 행동 경로. S_a는 시뮬레이터 반사실로 잰 행동 민감도다.
- 전체 미세조정: 같은 ID loss에서 S_a가 scratch의 0.90배다(시드 0.901~0.904). 같은 step에서는 1.02~1.07배이고, scratch S_a의 80%에 약 4배 빨리 도달한다. 사전학습 특징은 경쟁자가 아니라 발판이었다.
- 동결 backbone + AdaLN 행동: 같은 손실에서 0.57~0.60배. 정체값이 혁신 몫 1 − R²(a | 이력) ≈ 0.53과 비슷하다(관측 0.54~0.57).
- 가산 쌍둥이(사전학습 예측기 + 빈 행동망): 0.03~0.05배. 같은 구조에 학습 전 쌍둥이를 두면 0.95~1.01배다. 원인은 구조가 아니라 사전학습 선점이다. 에세이의 "learning in another model's shadow"에 해당한다.
- 미세조정 직후 같은 손실 지점에서 사전학습 모델의 S_a는 scratch의 0.08~0.14배다. ID loss가 행동 추종의 결손을 가린다.
- 한계: E1·E2·E2c는 사후 탐색이고, 장난감 규모이며, 4k step까지만 봤다(E2는 지연일 수 있다). 계획 효과는 재지 못했다.

### 3.2 가장 가까운 선행연구

| # | 논문 | URL | 이미 보인 것 | 확인 |
|---|---|---|---|---|
| 1 | Baldassarre et al., *Back to the Features: DINO as a Foundation for Video World Models* (DINO-world), arXiv 2507.19468 (2025-07) | https://arxiv.org/abs/2507.19468 ; https://arxiv.org/html/2507.19468 | 대규모 비디오로 행동 없이 사전학습한 DINOv2 latent 예측기에 행동을 붙이는 세 방식을 비교했다. scratch / 사전학습 동결 + action block만 학습 / 전체 FT의 계획 성공률(512 episode): PushT 46.9 / 49.4 / 59.4, Wall 87.1 / 91.1 / 93.8, PointMaze 59.4 / 61.6 / 68.7. 3.3절은 "the video world model can remain frozen, training only the action blocks"를 과적합 감소와 재사용 이점으로 제시한다. **같은 손실의 행동 민감도와 쌍둥이 구조는 없다. 동결이 scratch보다 계획을 잘했다는 결과는 C의 "동결은 굶는다"와 표면상 충돌한다** | [확인] |
| 2 | Sundar et al., *Action Forcing*, arXiv 2609.30595 (2026-09) | https://arxiv.org/abs/2609.30595 ; https://arxiv.org/html/2609.30595 | Wan2.1 기반 DiT를 행동 없는 비디오에서 복원한 egomotion 기저로 조건화한다. 4.1절 "GT continuation can masquerade as control": "GT agreement is high early because the pretrained model continues motion from the context." 주입 경로 ablation에서 AdaLN을 빼면 control 실패 58%, action token을 빼면 후진 실패 37%다. 같은 손실 비교는 없다 | [확인] |
| 3 | 행동 무시의 데이터 쪽 설명: Shi et al., *CoCo* (2608.04653); Zhang et al., *On the Identifiability of Controlled World Models* (2607.22430); Chen et al., WorldSync (2608.24885) | https://arxiv.org/abs/2608.04653 ; https://arxiv.org/html/2607.22430v2 ; https://arxiv.org/html/2608.24885 | 원인을 visual inertia라는 통계 편향(CoCo), excitation 부족(Zhang et al.), off-expert 지원 부족(WorldSync)으로 규정하고 반사실·IE 감독으로 고친다. WorldSync는 6개 backbone을 비교하지만 주입 방식이나 사전학습을 변수로 분리하지 않는다 | CoCo·Zhang [확인, R1], WorldSync [확인] |
| 4 | 사전학습 비디오 모델에 행동 붙이기: AVID (2410.12822), APV (2203.13880), PreLAR (ECCV 2024), DreamDojo (2602.06949), V-JEPA 2-AC (2506.09985), dWorldEval (2604.22152) | https://arxiv.org/html/2410.12822v1 ; https://arxiv.org/abs/2203.13880v1 ; https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03363.pdf ; https://arxiv.org/html/2602.06949 ; https://arxiv.org/html/2506.09985 ; https://arxiv.org/html/2604.22152v1 | AVID: 동결 비디오 확산 모델 출력에 학습된 mask와 adapter를 결합한다. APV: 행동 없는 사전학습 모델 위에 행동 조건 모델을 쌓는다. PreLAR: 행동 없는 사전학습과 행동 조건 최종 모델 사이의 gap이 사전학습의 효용을 제한한다. DreamDojo: 행동을 timestep embedding에 더해 AdaLN으로 넣고 전체를 미세조정한다. 행동 없는 사전학습의 이득은 "marginal"(PSNR 기준)이다. V-JEPA 2-AC: 인코더를 동결하고 새 300M predictor를 DROID 62시간으로 학습하며, 행동은 block-causal 토큰이다. dWorldEval: 행동을 시각 우세 denoiser의 보조 조건으로 넣으면 "strong visual priors to override control signals"라고 보고 행동과 시각을 같은 토큰으로 처음부터 학습한다 | AVID·APV·PreLAR [검색], DreamDojo·V-JEPA 2·dWorldEval [확인] |
| 5 | 주입 메커니즘 비교와 동결: ACWM-Phys (2605.08567), UWM (2504.02792), Nano World Models (2605.23993), DexAC (2606.27325), *Freeze, Share, Shrink* (2511.12101) | https://arxiv.org/abs/2605.08567 ; https://arxiv.org/pdf/2504.02792 ; https://arxiv.org/abs/2605.23993 ; https://arxiv.org/abs/2606.27325 ; https://arxiv.org/abs/2511.12101 | ACWM-Phys: cross-attention은 고차원 행동에서 이득이다. UWM: AdaLN을 cross-attention으로 바꾸면 나빠졌다. Nano World Models: 주입 방식을 요인으로 연구한다. DexAC: 국소 refinement와 전역 modulation을 결합한다. Freeze, Share, Shrink: diffusion policy에서 관측 없이 사전학습한 backbone을 동결하고 modulation 조건 경로만 학습하면 정상이고, attention 조건은 동결하면 붕괴한다. **동결 + modulation이 늘 나쁜 것은 아니다. 동결된 부분이 목표를 이미 설명하는 경쟁 경로가 아닐 때는 괜찮다는 대조 사례다** | UWM [검색], 나머지 [확인] |
| 6 | 학습 동역학 이론(개념적 선행): Pezeshki et al., *Gradient Starvation*, NeurIPS 2021; Huang et al., *Modality Competition*, ICML 2022; Wang, Tran, Feiszli, CVPR 2020; Peng et al., *OGM-GE*, CVPR 2022; Clark, Yatskar, Zettlemoyer, EMNLP 2019 | https://papers.nips.cc/paper/2021/hash/0987b8b338d6c90bbedd8631bc499221-Abstract.html ; https://proceedings.mlr.press/v162/huang22e.html ; https://arxiv.org/abs/1905.12681 ; https://openaccess.thecvf.com/content/CVPR2022/papers/Peng_Balanced_Multimodal_Learning_via_On-the-Fly_Gradient_Modulation_CVPR_2022_paper.pdf ; https://aclanthology.org/D19-1418/ | 먼저 잡힌 특징이 다른 예측 특징의 학습을 굶긴다(Pezeshki). late-fusion 다중 모달 네트워크를 함께 학습하면 modality가 경쟁하고 일부만 학습된다(Huang). 손실을 더 빨리 낮추는 modality가 최적화를 지배해 다른 쪽을 덜 학습시킨다(Wang; OGM-GE). bias만으로 예측하는 모델을 먼저 학습하고 주 모델을 그것과 ensemble로 학습하면, 주 모델이 bias로 설명되지 않는 패턴에 집중하게 된다(Clark et al.). **이 마지막 구조가 가산 쌍둥이와 같다. debiasing에서는 의도적으로 쓰는 구조가 월드모델에서는 행동 경로를 굶긴다** | [검색] |

**함께 볼 연구**
- Biderman et al., *LoRA Learns Less and Forgets Less*, TMLR 2024 [검색: https://arxiv.org/abs/2405.09673v2]. LoRA는 목표 도메인에서 전체 FT보다 덜 배우고 덜 잊는다. LoRA 부착의 예상 위치를 정하는 근거다.
- ControlNet (arXiv 2302.05543) [검색], ControlNet++ (arXiv 2404.07987, ECCV 2024) [검색]. 동결 원본 + 학습되는 복사 branch + zero convolution 설계와, 그 조건 추종이 부정확하다는 후속 보고. R2-10의 ControlNet식 branch 조건의 배경이다.
- Vid2World (arXiv 2505.14357, ICLR 2026) [검색], IRASim (arXiv 2406.14540, ICCV 2025) [검색]. 사전학습 비디오 확산을 인과화하고 행동 guidance를 더하거나, frame-level AdaLN 행동 조건을 쓴다.
- *Native Action-Prior Learning from Videos* (NAVA-WAM), arXiv 2610.03391 [확인]; A2World, arXiv 2606.29501 (ECCV 2026) [확인]. 행동 사전학습을 다른 방식으로 푼다. 부착 비교는 없다.
- AGRA, arXiv 2606.12217 [확인: 초록]. world action model에서 비디오 특징과 행동 decoder의 표현 불일치를 지적한다. 검색 요약에는 "비디오 전용 미세조정 뒤 backbone을 동결하면 성능 상한이 낮다"는 ablation이 있었다 [검색]. 정책 쪽 결과라 참고만 한다.

### 3.3 이미 보인 것과 남은 공백

**이미 보인 것**
- 행동 없이 사전학습한 비디오 prior가 행동 조건을 압도할 수 있다는 정성적 주장(dWorldEval, World-VLA-Loop, CoCo의 visual inertia, Action Forcing의 초기 맥락 연장)
- scratch, 동결, 전체 FT의 계획 성공률 비교(DINO-world)
- 주입 메커니즘(AdaLN, cross-attention, token)의 성능 비교(ACWM-Phys, UWM, Nano World Models, Action Forcing)
- 먼저 잡힌 경로가 다른 경로를 굶긴다(gradient starvation, modality competition), bias 모델의 그늘에서 배우기(ensemble debiasing)

**남은 공백**
1. 같은 ID loss에서 부착 방식별 행동 민감도를 반사실 정답으로 잰 연구는 없다. DINO-world는 계획 성공률만 보았고, 동결이 scratch보다 높았다.
2. 사전학습 쌍둥이와 학습 전 쌍둥이를 대조해 선점을 인과적으로 분리한 연구는 없다.
3. 동결·adapter의 정체값이 혁신 몫 1 − R²(a | 이력)를 따른다는 정량 예측은 없다.
4. "손실이 행동 추종을 가린다"를 같은 손실 기준으로 정량화한 연구는 없다. Action Forcing은 GT 일치로 정성적으로 관찰했다.
5. modality competition 이론을 행동 조건 월드모델의 부착 설계 지침으로 옮긴 연구는 없다.

### 3.4 판정: **중간**
- 기전(경쟁과 굶김)은 ML에서 잘 알려져 있다. 현상(시각 prior가 행동을 압도)도 정성적으로는 알려져 있다. 부착 방식 비교 자체도 DINO-world에 있다.
- 새로운 것은 같은 손실의 측정, 쌍둥이 대조, 혁신 몫 법칙, 그리고 그것을 실제 부착 설계(동결 adapter, ControlNet식 branch, 가산 branch)에 연결한 점이다.
- 지금 근거는 장난감 규모의 사후 탐색이다. R2-10의 사전등록 결과와 실제 사전학습 모델 1종 검증이 있어야 중간을 유지할 수 있다.

### 3.5 가장 위협적인 논문
**DINO-world (Baldassarre et al. 2025, arXiv 2507.19468).**
- Meta FAIR의 같은 계열 실험(행동 없이 사전학습한 DINOv2 latent 예측기에 행동 부착; scratch / 동결 / 전체 FT)이다.
- 결과도 "동결 > scratch"(계획)라서 리뷰어는 "DINO-world는 동결이 scratch보다 낫다고 했는데 왜 동결이 굶는다고 하나?"라고 물을 것이다. 같은 step 대 같은 손실, 계획 성공률 대 행동 민감도의 차이를 한 그림에서 정리해야 한다.
- 개념 쪽 위협은 modality competition 문헌(Huang 2022, OGM-GE)이다. "이것은 modality imbalance"라는 리뷰가 예상된다. 서론에서 먼저 연결하고, 월드모델에 고유한 점(혁신 몫, 반사실 정답, 계획 결과)으로 차별화한다.

### 3.6 포지셔닝

**제목 후보**
- *Where the Action Enters Decides the Race: Pretrained Pre-emption Starves Action Pathways in Video World Models*

**주장(한 문장)**
행동 없이 사전학습한 예측기에 행동을 붙일 때, 행동 추종은 부착 방식이 정한다. 사전학습 특징을 변조하며 전체를 미세조정하면 발판이 되고(같은 손실 0.9배, 약 4배 빠름), 동결·adapter는 이력이 예측하지 못하는 행동 몫(혁신 몫)에서 정체하며, 사전학습 출력에 더하는 별도 branch는 굶는다(0.03~0.05배). 굶김의 원인은 구조가 아니라 사전학습 선점이다.

**서술 원칙**
- modality competition과 gradient starvation의 월드모델판이라고 서론에서 밝힌다.
- CoCo와 Zhang et al.(데이터·식별성 설명)과의 차이를 명시한다. 세 부착 방식은 데이터와 excitation이 같다.
- DINO-world와의 관계를 정면으로 다룬다. 같은 step에서는 사전학습이 이기고(R2-2의 R = 1.02~1.07), 같은 손실에서는 동결이 지는 것이 함께 성립할 수 있다.

**안전하려면 필요한 실험**
1. **R2-10 완료.** LoRA rank, ControlNet식 복사 branch, twin과 twin0, 처방(twin_first, pdrop, lr10), lr sweep, 4배 학습으로 결손과 지연을 가른다. 판정은 사전등록대로 한다.
2. **실제 사전학습 모델 1종 이상.** DINO-world류 latent 예측기(가중치가 공개되어 있다면), V-JEPA 2 계열 predictor, 또는 Cosmos-Predict2 2B나 Wan 1.3B. 반사실 정답이 있는 시뮬레이터(Meta-World, ManiSkill, LIBERO)에서 전체 FT, 동결 + AdaLN, LoRA, 가산 branch를 비교한다. 지표는 같은 손실의 S_a, CoCo의 ARC·DE, WorldSync의 IE다.
3. **하류 효과.** 같은 모델로 계획 성공률과 정책 평가 순위 상관을 잰다. DINO-world의 "동결 > scratch(계획)"와 R2-2의 "동결 < scratch(같은 손실 S_a)"를 한 그림에서 조정한다. 행동 민감도가 필요한 과제(정밀 위치, 반사실 행동 평가)에서 차이가 커지는지 본다.
4. **혁신 몫 법칙의 조작 검정.** 혁신 분산을 고정하고 행동 자기상관만 바꿔(R1-2의 두 손잡이) 동결 정체값이 1 − R²를 따라 움직이는지 본다.
5. **이론 부록.** 선형-가우시안 모델에서 전체 FT, 동결, 가산 경쟁의 공 배분 동역학을 풀고, 정체값과 E2의 탈출 시간을 예측한다. Huang 2022, Pezeshki 2021과 연결한다.
6. **처방 기준선.** OGM-GE식 gradient modulation, CoCo·IE 반사실 손실, Action Forcing식 이중 주입(AdaLN + token).

**피해야 할 주장**
- "사전학습 비디오 prior는 행동을 굶긴다": 전체 FT에서는 반대(가속)였다.
- "adapter·LoRA·동결은 나쁘다": DINO-world 계획에서는 동결이 scratch보다 나았고, Freeze, Share, Shrink에서는 동결 + modulation이 정상이었다. "동결된 부분이 목표를 이미 설명할 때"라는 조건을 붙인다.

### 3.7 피지컬 AI 랩 어필
- **사전학습 비디오 모델을 행동 조건 월드모델로 바꾸는 그룹**: Cosmos 계열(NVIDIA)과 DreamDojo, Tsinghua(Vid2World; PDF가 Tsinghua의 ~mlong 페이지에 있음), Microsoft Research(AVID, 검색 결과 기준), Meta(DINO-world, V-JEPA 2-AC), Stanford·Tsinghua(Ctrl-World), AgiBot(Genie Envisioner).
- 실무 메시지: 손실 곡선 대신 S_a 같은 반사실 지표를 모니터하라. 가산 branch나 동결 adapter를 쓸 때는 정체값이 혁신 몫에 갇혔는지 확인하라.
- **ML 이론 쪽(학습 동역학, multimodal imbalance)**: 생성형 월드모델이라는 새 시험대와 반사실 정답이라는 깨끗한 측정을 준다.

---

## 4. 예비 축

### 4.1 D — Crushing a visual world model
- **가설**: 시각 월드모델 예측기의 폭을 줄이면 물리 변수의 해상도는 떨어지지만 무관한 변수끼리 섞이지는 않는다(coarsening, not packing). 좁은 구분(접촉 직전 대 접촉)이 먼저 사라진다.
- **가까운 연구**
  - 원문 에세이의 crushing 실험(비전 분류기)
  - Elhage et al., *Toy Models of Superposition*, 2022 [검색: https://arxiv.org/abs/2209.10652v1]. 차원이 부족하면 덜 중요한 특징부터 버리고, 희소할수록 superposition으로 packing한다.
  - Kang et al., *How Far is Video Generation from World Model*, ICML 2025 [검색: https://arxiv.org/abs/2411.02385v1]. 비디오 생성모델은 case-based로 일반화하고, 참조 우선순위는 color > size > velocity > shape다.
  - Hooker et al. 2019 [검색, R1]. 압축은 어렵고 비전형적인 예부터 잊는다.
  - Fu et al., Ranganath et al. [확인, R1]. 비트 압축이지 폭 압축이 아니다.
  - Tan et al., PokeWorld (arXiv 2607.27017) [확인, R1]. latent 월드모델이 숨은 물리 변수를 담는지와 쓰는지를 나눴다.
  - WestWorld (arXiv 2603.14392) [검색]. 큰 trajectory 월드모델을 2블록 student로 증류해 MPPI에 쓴다. 무엇을 잃는지는 분석하지 않았다.
- **공백**: 월드모델 폭 sweep에서 혼동(packing) 지표와 뉴런 국소성으로 coarsening과 packing을 가른 연구는 찾지 못했다.
- **판정: 중간.** 시험 자체는 새롭다. 그러나 "작은 모델은 미세 구분을 잃는다"가 자명하게 들릴 위험과 null 위험이 크다. R2-11 결과는 아직 없다.
- **조언**: 계획 영향(정밀 목표 대 거친 목표)과 묶어야 피지컬 AI 논문이 된다. 단독보다 축 A(압축과 계획)의 부록이나 해석 논문(R1-5류)에 붙이는 편이 낫다. StefanHex의 지적대로 중간 폭을 촘촘히 sweep한다.

### 4.2 E — CEM 반복에 따른 순위 붕괴와 해상도 인지 계획
- **파일럿**: DINO-WM에서 후보 비용과 참 비용의 Spearman이 PushT 0.906 → 0.529, Wall 0.918 → 0.510으로 떨어졌다.
- **가까운 연구**
  - Wang, Zuo, Rui, Feng, Li, *Decision-Metric Alignment in Latent World Models*, arXiv 2608.18746 [확인: https://arxiv.org/html/2608.18746]. PushT에서 CEM 단계(random / mid / elite)별 Spearman이 LeWM 기준 +0.403 / +0.227 / +0.036이다. 모든 변형에서 elite 단계가 0 근처다("shared local saturation"). 후보 margin이 "can shrink further as CEM concentrates its proposal"이라고 분석한다.
  - Obst & Stolzenburg, arXiv 2610.00921 [확인]. iteration이 진행돼도 elite 일치가 좋아지지 않고 proposal은 수축한다.
  - Zhang, Shi, Peng, Xie, *Knowing When to Stop Thinking*, arXiv 2610.28011 (2026-10-20) [확인: https://arxiv.org/html/2610.28011v1]. DINO-WM Wall에서 성공률이 iteration 11에 0.576으로 정점을 찍고, 예측 비용이 계속 떨어지는 동안 iteration 30에 0.356으로 38.2% 떨어진다. elite 비용 정체로 멈추는 규칙은 plateau 과제에는 통하지만 정점 이후의 harmful refinement는 막지 못한다.
  - ARC-Bench [확인]; Alrasheed, Azam, Akhtar, *The Planning Limits of Latent World Models*, arXiv 2609.39235 [확인: https://arxiv.org/abs/2609.39235]. 상상 궤적 범위 안의 목표에서만 순위가 믿을 만하다.
- **남은 공백**: 범위 축소(range restriction)를 고정 잡음 oracle로 걷어낸 분석과 해상도 비 R 정도다. 폭이 좁다.
- **판정: 낮음.** 단독 논문은 권하지 않는다. 축 A의 iteration 축 증거로 흡수한다. 2610.28011의 "예측 비용은 계속 떨어지는데 성공은 떨어진다"는 축 A의 기전(반복 최적화기가 지속 오차를 착취)과 같은 현상으로 인용할 수 있다.

---

## 5. 종합 권고

- **노벨티 기준 우선순위: A > C > B.** 시의성은 B가 가장 높지만 경쟁도 가장 심하다.
  - **A**: 가장 독특한 실험 결과다. 같은 err1에서 공유 범위만 바꿔 하락이 5%p에서 85%p까지 갈리고, 값싼 처방이 있다. 고전 이론과 연결해 깊이를 보일 수 있다. 약점은 DMC 효과가 작다는 점, V-JEPA 반복 CEM이 미검증이라는 점, 체크포인트가 하나라는 점이다. 1.6의 1~3번을 먼저 한다.
  - **C**: 기전이 가장 깨끗하다(쌍둥이 대조). 하지만 장난감 규모이고 개념상 modality competition과 가깝다. R2-10과 실제 사전학습 모델 1종 검증이 필수다.
  - **B**: 현상은 붐빈다. 정책 수준 법칙과 공개 모델 검증(R2-12)이 성패를 가른다. CureWM 저자들이 정책 순위를 다음 과제로 남겼으므로, B를 고른다면 빨리 움직여야 한다.
- **합치기**
  - A + E: E를 A의 iteration 축 증거로 넣는다.
  - C + R1-2(두 손잡이): "racing은 언제 결손이 되는가"라는 한 편의 논문이 된다.
  - D는 A의 부록 후보다.
- **지원서 서사**: "월드모델의 오차는 크기보다 구조가 결정을 정한다"는 하나의 주제로 세 축을 꿸 수 있다.
  - A: 계획기 안에서의 오차 구조(후보 간 공유, iteration 간 지속)
  - B: 결과 공간의 경계 근처에서의 오차(near-miss)
  - C: 학습 동역학이 만든 행동 경로의 결손(부착 위치)

---

## 6. 검색했지만 찾지 못한 것 (공백 주장의 근거, 제출 직전 재확인 필요)

- 학습된 월드모델의 압축 오차에 CRN이나 dither를 적용해 계획을 지킨 연구(월드모델과 VLA 모두)
- MPPI/CEM에서 모델 오차의 iteration 간 지속성을 조작한 연구. 가장 가까운 것은 고전 SA/CRN과 Ha & Schmidhuber의 temperature 사례다.
- 실제 로봇 월드모델(V-JEPA 2-AC 등)에서 결정 near-tie 밀도와 정밀도 flip을 잰 연구. 가장 가까운 것은 LLM 비결정성 연구와 DINO-WM fp16 집계 성공률이다.
- 월드모델 정책 평가 편향을 정책의 near-miss 비중의 함수로 분석한 연구. CureWM은 policy ranking을 미검증으로 명시했다.
- 경계 근처(작은 severity)까지 촘촘히 잰 실패 거리별 상상 성공 곡선
- 같은 ID loss에서 부착 방식별 행동 민감도를 비교하고, 사전학습 쌍둥이와 학습 전 쌍둥이를 대조한 연구
- 동결·adapter 행동 부착의 정체값을 혁신 몫으로 예측한 연구
- 월드모델 폭 crushing에서 packing과 coarsening을 가른 연구
- 원문 에세이(2026-10-15)를 로봇, 월드모델, 계획으로 확장한 후속 연구(2026-10-21 기준)

## 7. R1 검토 정정과 보충

- **WorldOracle (arXiv 2610.18826)**: R1은 "causal obedience를 측정했다"고 적었다. 이번에 연 초록(v2, 2026-10-16)에는 그 용어가 없다. 실제 내용은 같은 초기 관측과 저수준 행동열을 넣어 실제 실행(성공과 실패 포함)과 비교하는 ACWM 벤치마크다. 4개 차원(물리 상태 정확성, 상호작용 공간 정렬, 핵심 상호작용 영역 충실도, 다시점 행동 일관성)으로 7개 모델을 평가한다.
- **MiraBench**: optimism bias의 교란 severity는 s = 0.5 하나다(부록 B). near-miss와 far-miss의 비교와 정책 순위 분석은 없다. R1의 "성공 prior × 잔차 행동항" 가설은 본문의 "decoupling hypothesis"와 같다.
- **ARC-Bench**: 초록 기준으로 V-JEPA 2-AC 자체를 평가했는지는 확인되지 않는다. backbone 확장에서 V-JEPA 1·2 인코더(ViT-L, ViT-G)를 썼다.
