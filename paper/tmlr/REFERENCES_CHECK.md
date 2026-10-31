# 참고문헌 확인 기록 (TMLR 원고)

- 작성: 2026-10-31. 서브에이전트가 arXiv API 기록(초록 페이지와 같은 데이터)과 공식 proceedings 페이지(PMLR, ICLR 2026, NeurIPS, CVF, RSS, TMLR, Crossref, CoRL 2026 채택 목록)로 제목, 전체 저자, 연도, 출판처를 확인했다. arXiv와 출판본이 다르면 출판본을 따랐다.
- 결과: [`refs.bib`](refs.bib) 59항목. 58항목은 확인 완료. Holm(1979)은 JSTOR가 자동 접근을 막아 제목만 JSTOR 기록으로 확인했고 권·쪽은 2차 출처를 따랐다(DOI 10.2307/4615733이 404라 JSTOR URL만 넣음). 확인하지 못한 항목은 bib에 넣지 않았다.
- 원고는 이 가운데 인용한 항목만 참고문헌에 나온다(bibtex).

## 1. 기존 인용에서 바뀐 점

| 항목 | 바뀐 점 |
|---|---|
| DINO-WM | arXiv 2024가 아니라 ICML 2025(PMLR 267)로 인용한다. 키 `zhou2025dinowm` |
| Cosmos Policy, Ctrl-World, WorldGym, WMPO | ICLR 2026 |
| DreamDojo | ICML 2026(PMLR 저자 23명, arXiv는 30명) |
| dWorldEval | ICML 2026에 다른 제목 "Scaling Robot Policy Evaluation via Discrete Diffusion World Model"로 실림 |
| FACT, WoVR | CoRL 2026 채택 |
| Completion Aware Guidance | NeurIPS 2026 본 학회가 아니라 워크숍 포스터(arXiv 2610.01559) |
| Diffusion Policy | RSS 2023판(저자 7명)으로 인용한다. arXiv v5는 IJRR 2025 확장판이다 |
| Push-T | 원출처는 IBC(Florence et al., CoRL 2021). IBC와 Diffusion Policy를 함께 인용한다 |
| Qwen2.5-VL | 기술 보고서는 3B·7B·72B만 다룬다. 32B는 model card로 따로 인용한다 |
| SIMPLER | PMLR 저자 순서를 따른다. MMRV 정의는 원문에서 확인했다 |
| Veo 평가기 | 예측 절대 성공률이 실제보다 낮다는 점을 원문에서 확인했다(설계 문서의 "원문 미확인"을 해소). 원고 2절에 썼다 |
| RoboWorld | VLM 채점표에 "near success" 등급이 있다. "near-miss로 층화한 분석이 없다"는 표현은 쓰지 않았다 |

## 2. 새로 찾은 관련 연구 (arXiv 2026-09-01 ~ 10-31)

- 직접 경쟁 연구(실패 거리로 색인한 상상 오판, 또는 near-miss 비중과 평가 편향의 관계)는 없다. 10-24 판단과 같다.
- 가장 가까운 것: **WorldFailBench** (arXiv 2610.33565, 10-24). 7개 월드모델이 이웃 과제 조건 사이의 성공률 하락을 과소 추정한다(응답 420개 중 400개). 원고 2절에서 차이를 밝혔다: closed-loop 평가이고, 실패 거리 색인, 판정기·상상 분리, near-miss 비중과 편향의 관계가 없다.
- 원고에 인용한 새 연구: WorldGuide(2609.34206), AIM(2610.27387, CoRL 2026), TEMPO(2610.04988)
- bib에만 넣은 것(필요하면 인용): Physical Canvas, DreamAvoid, When World Models Lie, DRILL, MEND, AnyRank, RoboFQE, DLS, StressDream, Foresight, No Free Checker, WorldOracle, NeurIPS 2026 워크숍 2편
- 투고 직전에 10-30 이후 제출분을 다시 검색한다(색인 지연).

## 3. 넣지 않은 것

- Dreamer, TD-MPC2, Ha & Schmidhuber: B 설계 문서가 인용하지 않는다.
- π0.5, OpenVLA-OFT: M2 정책 단계를 실행하지 않았다.
- AMD 기술 글 "Learning to Fail…": 페이지가 열리지 않아 확인하지 못했다.
