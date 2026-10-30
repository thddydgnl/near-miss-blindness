# JKAIA 2026 EasyChair 입력값 — B (Near-Miss Blindness)

- 학회 규정: 논문은 영문, EasyChair의 **제목은 영문, 초록은 국문**으로 입력한다(논문제출 안내, 2026-10-24 확인).
- 아래 국문 초록은 `main_ko.tex`의 초록과 같은 글이고, 영문 `main.tex` 초록과 문장 단위로 대응한다.
- 템플릿 규칙(제목·초록에 기호, 특수문자, 각주, 수식 금지)을 국문 초록에도 그대로 지켰다. 쓴 문자는 한글, 영문, 숫자, 소수점, 쉼표, 괄호, 하이픈뿐이다.
- 수치는 모두 `main/results`의 JSON과 대조했다(`paper/build/verify_numbers.py`).
- 2026-10-24 독립 검토를 반영한 판이다: 근거 칸 한정(상상 길이 5단계), "정보의 일부"로 완화, "시험한 두 latent 모델"로 범위 한정.

- 2026-10-25 2차 검토 반영: 처방 문장에 "두 latent 모델 모두에서"를 넣어 범위를 바로잡았다(영문 초록 "in both latent models"와 대응).

- 2026-10-25 권장 항목 반영: 제목을 "Loses Resolution at the Outcome Boundary"로 완화했고, 초록에 내부 사전등록(시각·해시 기록 설계 문서)을 명시했으며, "경계는 잃는다"를 "경계의 해상도는 잃는다"로 바꿨다. 짧은 초록도 같은 뜻으로 고쳤다.

## EasyChair 입력 항목

| 항목 | 입력값 |
|---|---|
| Track | **General (Poster)** |
| Title | Near-Miss Blindness: Latent World-Model Imagination Loses Resolution at the Outcome Boundary |
| Authors | YongHwi Song, 충북대학교 정보통신공학부(School of Information and Communication Engineering), thddydgnl1937@gmail.com (교신) / Keon Myung Lee(이건명), 충북대학교 소프트웨어학부(School of Software), kmlee@cbnu.ac.kr |
| Keywords (한 줄에 하나) | World models / Robot policy evaluation / Failure prediction / Imagination / Pre-registration |
| Abstract | 아래 "국문 초록" |
| Paper | `main.pdf` (영문, 2쪽) |

## 국문 초록 (붙여 넣을 글, 802자·공백 제외 623자)

```text
로봇 정책 평가에 쓰이는 학습된 월드모델은 실패한 실행을 성공으로 상상한다고 보고되어 왔다. 본 연구는 어떤 실패가 오판되는지를 묻는다. 성공 경계 주변에 반사실 실패를 촘촘하게 만들어 실패 거리(과제 오차를 성공 문턱으로 나눈 값)로 정렬하고, 실제 최종 프레임과 상상 최종 프레임을 같은 판정기로 판정해 판정기 자체의 오차와 상상이 더한 오차를 분리한다. 가설, 문턱, 판정 규칙은 시각과 해시를 기록한 내부 설계 문서로 사전등록했다. 공개 DINO-WM PushT 모델에서 상상 길이가 5단계일 때 상상은 성공과 먼 실패를 여전히 구분하지만 경계의 해상도는 잃는다. 성공 대 near-miss AUROC는 실제 프레임의 0.953에서 상상의 0.853으로 떨어지는 반면 성공 대 far-miss AUROC는 0.005만 떨어지며, near-miss 쪽 하락은 상상 길이와 함께 커진다. 상상 특징으로 학습한 판별기도 0.889에 그쳐 경계 정보의 일부가 사라지며, 오판은 낙관 쪽이 아니라 양방향으로 일어난다. 통제된 latent 모델에서 학습 데이터의 실패율을 0.3에서 0.03으로 낮추면 near-miss를 성공으로 상상하는 비율이 0.10에서 0.54로 늘고 이 오판은 모두 상상에서 생기며, 성공을 실패로 보는 오판은 0으로 줄어든다. 공개 모델에서 재학습 없는 MC dropout 할인의 효과는 작았고, 두 latent 모델 모두에서 경계를 겨냥한 반사실 데이터는 단순 재가중보다 낫지 않았다. 비디오 월드모델에서는 사전등록한 경계 특이성 가설이 지지되지 않아, 주장의 범위를 시험한 두 latent 모델로 한정한다.
```

## 짧은 국문 초록 (초록 칸에 글자 수 제한이 있을 때, 422자)

EasyChair 제출 화면은 로그인해야 보여서 글자 수 제한을 미리 확인하지 못했다. 제한이 있으면 이 글을 쓴다.

```text
월드모델 안에서 로봇 정책을 평가할 때 어떤 실패가 성공으로 오판되는지 내부 설계 문서로 사전등록한 실험으로 조사했다. 성공 경계 주변에 반사실 실패를 촘촘하게 만들고 실제 프레임과 상상 프레임을 같은 판정기로 판정했다. 공개 DINO-WM PushT 모델에서 상상 길이가 5단계일 때 성공 대 near-miss AUROC는 0.953에서 0.853으로 떨어지지만 성공 대 far-miss AUROC는 0.005만 떨어져, 상상은 경계 근처의 판별력을 잃는다. 통제된 latent 모델에서는 학습 실패율을 0.3에서 0.03으로 낮추면 near-miss를 성공으로 상상하는 비율이 0.10에서 0.54로 늘어, 성공 편중 데이터가 경계 흐림을 낙관으로 바꾼다. 비디오 월드모델에서는 이 양상이 지지되지 않아 주장을 시험한 두 latent 모델로 한정한다.
```

## 용어 대응 (국문 초록 ↔ 영문 원고)

| 국문 | 영문 |
|---|---|
| 실패 거리 | failure distance r = e / τ |
| near-miss / far-miss | 1 ≤ r < 2 / r ≥ 3 |
| 판정기 | success judge (주 판정기는 oracle 분류기, judge (a)) |
| 상상 특징으로 학습한 판별기 | imagined-feature probe (정보 상한) |
| 오판은 양방향 | the errors are two-sided |
| 실패 노출량을 맞추면 | at matched (failure) exposure |
| 경계 특이성 가설 | pre-registered boundary-specific hypotheses (H1a·H1b) |
