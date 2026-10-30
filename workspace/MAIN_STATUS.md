# 본 실험 진행 상황 (조정자가 갱신)

- 시작: 2026-10-21T13:00Z
- 규칙: [MAIN_RULES.md](MAIN_RULES.md)
- RunPod: 사용자 상한 $60. 감시 스크립트가 $55에서 모든 pod를 지운다. 장부는 `runpod_main/pods.ledger`이고, 시작 잔액은 $120.01이다.
- 서버 GPU 배정: GPU 0은 A, GPU 1은 B(M1·M3)

## A 단계

| 단계 | 상태 | 장소 | 보고 |
|---|---|---|---|
| E0 TD-MPC2 쪽(환경, mujoco-py, 재현, 과제·k 선정) | **완료**(18:09Z, mujoco-py 성공, RunPod 불필요) | 서버 GPU 0 | [E0_tdmpc2.md](A_dithered_world_models/main/reports/E0_tdmpc2.md) |
| E0 DINO-WM 쪽(환경, 재현, k 선정) | **완료**(19:58Z, R2-13 재현 일치, PointMaze 동작) | 서버 GPU 1 | [E0_dinowm.md](A_dithered_world_models/main/reports/E0_dinowm.md) |
| DESIGN_main.md v2(DINO-WM 세부) | **덧붙임**(20:00Z). PushT eval 쌍과 OEI 기준은 사용자 확인 뒤 v3 | — | |
| E1 DINO-WM Wall·PointMaze(280 job, Wall에 k=3 탐색 칸 포함) | **완료**(13:15Z, 누락 0) | 서버 GPU 1 | |
| E1 DINO-WM PushT | **완료**(18:56Z) | 서버 GPU 0·1 | |
| E1 DINO-WM 분석·보고 | **완료**(19:40Z). H1 성립, H3a 성립(4/5, 해로운 칸 없음), H2a 전체 불성립(PushT만 성립), H2b 불성립. **A go/no-go = go**(TD-MPC2 MW와 DINO-WM H3a 모두 성립) | 맥북 | [E1_dinowm.md](A_dithered_world_models/main/reports/E1_dinowm.md) |
| DESIGN_main.md | **v1 고정**(18:12Z: 가설·판정 규칙 전체 + TD-MPC2 세부). DINO-WM 세부는 v2로 덧붙임 | — | [DESIGN_main.md](A_dithered_world_models/main/DESIGN_main.md) |
| E1 TD-MPC2(12과제 × 칸 3 × 7변형 × 90 ep, 4,320 job) | **진행**(18:14Z 시작). episode 4,104개 완료(06:08Z), 칸 단위 job은 06:20Z부터 MPS 없이 다시 돌림 | 서버 GPU 0 | |
| DESIGN_main.md v4(E1 판정 계산 세부, 두 계열 공통) | **덧붙임**(06:55Z, TD-MPC2 E1 결과와 OEI를 보기 전) | — | |
| E1 TD-MPC2 분석·보고 | **완료**(07:40Z). H1 성립. H3a는 Meta-World 성립(19/19), DMC 불성립(walker k=3 해로운 칸). H2a·H2b 불성립. go/no-go는 DINO-WM 결과 대기 | 서버 CPU | [E1_tdmpc2.md](A_dithered_world_models/main/reports/E1_tdmpc2.md) |
| E4 이론·모의 | **완료**(09:15Z, CPU 11.6시간). 유도와 모의가 잘 맞음. 결정 사항 3건 중 H2c elite 처리와 E3 var_cand는 v5.1에 반영 | 서버 CPU | [E4_theory.md](A_dithered_world_models/main/reports/E4_theory.md) |
| DESIGN_main.md v5·v5.1·v5.2(E2·E3·E5·E7 실행 세부) | **고정**(v5 09:00Z, v5.1 09:30Z: E5 α 격자, H2c는 elite 수 고정 sweep, E3 var_cand; v5.2 09:50Z: 사용자 승인 사항) | — | [DESIGN_main.md](A_dithered_world_models/main/DESIGN_main.md) |
| E2·E3·E5·E7 | **실행 완료**(10-24 14:41Z). TD-MPC2는 H2a 재측정, E2, E3, E5, E7 성공률·지연(10-24 02:15Z), DINO-WM은 E2 N, E3, E5, E2 EO(14:41Z)까지 모두 실패 0. 남은 분석: DINO-WM E2 EO(기술), 사후 점검 정리, A 최종 보고 | 서버 | |
| E5 TD-MPC2 분석·보고 | **완료**(22:27Z). **H3b의 TD-MPC2 몫 성립(경계)**: 정확히 10/19칸(Meta-World 6/12, DMC 4/7). QAT 보고 조건은 0/11칸으로 걸리지 않았지만, QAT+det가 11칸 모두 B+det보다 나빠 약한 기준선으로 보인다(처리 방법은 사용자 확인). 조정자가 JSON으로 다시 세어 일치 | 서버 CPU | [E5_tdmpc2.md](A_dithered_world_models/main/reports/E5_tdmpc2.md) |
| E3 DINO-WM 분석·보고 | **완료**(22:39Z). **H4a의 DINO-WM 몫 성립(경계)**: 자격 2과제 모두 성립. PushT k=4 ρ −0.80(이웃 자리바꿈 한 번), PointMaze k=3 −1.00. Wall k=5는 비자격(기술, −0.20). 2축 대조는 사전 예상(indplan ≈ ind)과 맞음. 조정자가 job JSON으로 다시 계산해 일치 | 서버 CPU | [E3_dinowm.md](A_dithered_world_models/main/reports/E3_dinowm.md) |
| E7 TD-MPC2 분석·보고 | **완료**(02:30Z). **H5 성립**(네 읽기 모두): dither 지연 오버헤드는 12조건 모두 10% 이하(int8 최대 +3.8%, int4 +5.8~6.9%). Pareto 지배는 mw-push 사례뿐이고 약하다(같은 반복 수에서는 지연 차가 잡음 수준). 정수 kernel이 fp32보다 약 18% 느려 낮은 비트의 지연 이득은 없다. 조정자가 JSON으로 확인 | 서버 CPU | [E7_tdmpc2.md](A_dithered_world_models/main/reports/E7_tdmpc2.md) |
| E5 DINO-WM 분석·보고 | **완료**(12:29Z). **H3b의 DINO-WM 몫 불성립**(1/3칸: PointMaze k=3만 통과, PushT는 최선 PTQ만으로 det 손상이 거의 사라짐). QAT 규칙 조건은 걸리지 않음(QAT+det CI 상한 13%p). QAT+det는 최선 PTQ와 비슷해, TD-MPC2식 '약한 기준선' 단서는 달지 않고 그렇게 기술(조정자, v5.3 §10.1 적용). 가중치 SR은 RTN보다 나쁨(TD-MPC2와 반대). 조정자가 job JSON으로 다시 계산해 일치 | 서버 CPU | [E5_dinowm.md](A_dithered_world_models/main/reports/E5_dinowm.md) |
| DESIGN_main v5.3(사후 점검 두 가지, QAT 보고 방식) | **고정**(10-24 02:38Z, 사용자 승인) | — | [DESIGN_main.md](A_dithered_world_models/main/DESIGN_main.md) |
| 사후 점검: TD-MPC2 QAT+none, E7 지연 재측정 | **완료**(16:39Z, 사후·기술, 판정 변경 없음). 지연: 짝 오버헤드 중앙값 36조건 모두 ±3.2% 안, CI는 35/36이 0을 포함. 다만 측정 잡음(block 두 모드) 때문에 조건별 상한 < 10%는 보이지 못함(16/36이 넘음). QAT+none: 11/11칸에서 원래 모델보다 크게 나쁨(QAT 학습 자체의 손상). 조정자 재계산 일치 | 서버 GPU 0 | [posthoc_tdmpc2.md](A_dithered_world_models/main/reports/posthoc_tdmpc2.md) |
| A 최종 보고(FINAL_A.md) | **초안 완료**(10-24 22:33Z). 판정 전체 표, 최종 주장, 한계, 결정 항목 13개(추천안과 선택지, §7.14 요약). JKAIA 포스터 원고(`paper/jkaia2026/`, 10-24 다른 세션에서 작성, 마감 10-25)에서 결과와 어긋나는 문장 두 곳을 찾아 사용자에게 알림 | 맥북 | [FINAL_A.md](A_dithered_world_models/main/reports/FINAL_A.md) |
| E2·E3 TD-MPC2 분석·보고 | **완료**(16:05Z). H2a 재현 0/3, H2c 3/3 성립, (b1) 1/3, (b2) H2a′ 3/3 성립. H4a TD-MPC2 몫 3/4 성립(경계: 두 과제가 정확히 −0.8 = 이웃 자리바꿈 한 번), H4b 불성립(ρ +0.03), DMC 설명 "설명됨"(ρ +0.83). 조정자가 결과 JSON으로 다시 계산해 일치 | 서버 CPU | [E2_tdmpc2.md](A_dithered_world_models/main/reports/E2_tdmpc2.md), [E3_tdmpc2.md](A_dithered_world_models/main/reports/E3_tdmpc2.md) |
| E2·E3 계열 합산 판정(조정자) | **H2c 성립**(3/4, PushT만 불성립). **H2a′ 성립**(4/4). H2a 재현 불성립(1/4), H2a_indep 불성립(2/4). H4b 불성립(보조 15과제 −0.32/+0.21). **H4a 성립**(두 계열 모두, 22:40Z. 성립 5과제 중 3과제가 정확히 −0.8인 경계). DMC는 v5.2 §9.2대로 처방 주장에서 빼고 S_det로 서술. 기전 문장 문구와 처방 범위("반복하는 계획기")는 A 최종 보고 때 사용자에게 묻는다 | 맥북 | [LOG.md](A_dithered_world_models/main/LOG.md) |
| E2 DINO-WM 분석·보고(N sweep, elite·opt_steps) | **완료**(15:55Z). **H2c의 PushT 몫 불성립(경계)**: ρ_s(N, \|OEI_det\|) +0.943, ρ_s(N, D_det) +0.771(< 0.8). D_det는 N ≥ 300에서 평탄하다. 조정자가 job JSON으로 다시 계산해 일치. H4b 보조용 세 과제 값도 냈다. E2 EO 절 완료(16:38Z, 기술): 반복 수가 det 손상을 가장 크게 움직인다(5·10·30회에서 +7·+17·+24%p). elite 비율과는 단조 관계가 없고, sh-iter 이득은 모든 설정에서 남는다. 조정자 재계산 일치 | 서버 CPU | [E2_dinowm.md](A_dithered_world_models/main/reports/E2_dinowm.md) |
| E6(선택) | 대기 | 서버 | |

## B 단계

| 단계 | 상태 | 장소 | 보고 |
|---|---|---|---|
| E0 서버(M1·M3 환경과 재현, PushT DP 학습) | **완료**(15:40Z) | 서버 GPU 1 | [E0_server.md](B_near_miss_blindness/main/reports/E0_server.md) |
| E0 RunPod(Cosmos Policy·CureWM 재현, LIBERO 과제와 e 정의) | **완료**(14:48Z, $0.87, pod gone) | RunPod | [E0_runpod.md](B_near_miss_blindness/main/reports/E0_runpod.md) |
| E0b RunPod(LIBERO DP 학습, 판정기 (a)·(d) 재시도, severity 보정, 여러 pod용 runner) | **완료**(20:40Z, $5.82). (a) held 0.967로 M2 주 판정기, (d)는 보조 | RunPod | [E0b_runpod.md](B_near_miss_blindness/main/reports/E0b_runpod.md) |
| DESIGN_main.md | **v1 고정**(20:50Z). M3 n_π 구간만 보류(v2) | — | [DESIGN_main.md](B_near_miss_blindness/main/DESIGN_main.md) |
| E1 M2(곡선, 4,184 + t0 468) | **완료**(01:00Z, $13.58, pod gone). **사전등록 규칙상 M2 no-go**: H1a(near 0.298, near/far 2.52)와 H1b(성공 vs far ΔAUROC 0.137)가 모두 불성립 | RunPod A40 | [E1_m2.md](B_near_miss_blindness/main/reports/E1_m2.md) |
| E1 M1·M3(곡선) | **완료**(23:30Z). M1: H1a 불성립(near 0.147), H1b 성립. M3: H1a 성립(near 0.535), H1b의 두 조항 성립(horizon 해당 없음) | 서버 GPU 1 | [E1_server.md](B_near_miss_blindness/main/reports/E1_server.md) |
| E2·E3 서버(M1 DP 24개 정책 법칙, M3 정책, 판정기 분해) | **완료**(03:00Z). M1: H2c만 성립(H2a 0.357). M3: H2a 성립(0.829)과 H3 성립. M1 H3 불성립 | 서버 GPU 1 | [E2E3_server.md](B_near_miss_blindness/main/reports/E2E3_server.md) |
| E4·E5 서버(M1·M3 기전과 처방) | **완료**(11:40Z). H4a 성립(M3; M1 주 칸 불성립), H4b M1 성립·M3 불성립, H5a 불성립(0/3), H5b 성립(작음) | 서버 GPU 1(A와 공유) | [E4E5_server.md](B_near_miss_blindness/main/reports/E4E5_server.md) |
| B 최종 보고(전 단계 종합) | **완료**(13:05Z). H3 판정 보충(M2·M3 성립, 사용자 승인 12:55Z) 반영 | 맥북 | [FINAL_B.md](B_near_miss_blindness/main/reports/FINAL_B.md) |
| M2 E2, E6 | **생략**(사용자 승인, 주장 좁힘) | — | |

## 1시간 넘게 걸리는 작업

| 이름 | 장소 | 시작 | 예상 종료 | 로그 | 상태 |
|---|---|---|---|---|---|
| **A-E1-tdmpc2** (E1 TD-MPC2, 4,320 job) | 서버 GPU 0, tmux `A-E1-tdmpc2` | 18:14Z(19:11Z 분할 재시작) | 06:09Z | `~/pmx/A/logs/E1_tdmpc2.log` | episode 단계 완료, 칸 단위 job 실패(아래 줄로 다시 돌림) |
| **A-E1-tdmpc2-cells** (칸 단위 job 95개, MPS 없음) | 서버 GPU 0, tmux `A-E1-tdmpc2-cells` | 06:20Z | 약 07:00Z | `~/pmx/A/logs/E1_tdmpc2.log` | 진행 |
| **A-E1-dinowm-pusht** (PushT E1) | 서버 GPU 0 `A-E1-dinowm-pusht-g0`, GPU 1 `A-E1-dinowm-pusht-g1` | 06:08Z | 18:56Z | `~/pmx/A/logs/E1_dinowm_pusht_g*.log` | 완료 |
| **A-v5-queue-g0** (TD-MPC2 H2a 재측정 → E2 → E3 → E5 → E7 성공률·지연, MPS) | 서버 GPU 0, tmux `A-v5-queue-g0` | 10-22 19:05Z | 10-24 02:15Z | `~/pmx/A/logs/v5_queue_g0.log`, `v5_<단계>.log` | **완료**(QUEUE_DONE, 모든 단계 실패 0, MPS 멈춤) |
| **A-v5-dinowm-g1** (DINO-WM E2 N → E3 → E5 → E2 EO) | 서버 GPU 1, tmux `A-v5-dinowm-g1` | 10-22 19:04Z | 10-24 14:41Z | `~/pmx/A/logs/v5_dinowm_g1.log` | **완료**(ALL_DONE, 실패 claim 없음) |
| **A-v5-dinowm-g0** (GPU 0이 남은 DINO-WM 칸을 나눠 맡음, `plans/v5_g0.txt`) | 서버 GPU 0, tmux `A-v5-dinowm-g0` | 10-24 02:17Z | 10-24 14:16Z | `~/pmx/A/logs/v5_dinowm_g0.log` | **완료**(ALL_DONE) |
| **A-posthoc-g0** (사후 점검: QAT+none → MPS 멈춤 → 지연 재측정, v5.3) | 서버 GPU 0, tmux `A-posthoc-g0` | 10-24 14:24Z(시작) | 10-24 16:26Z | `~/pmx/A/logs/posthoc_g0.log`, `posthoc_qatnone.log`, `posthoc_e7lat.log` | **완료**(POSTHOC_DONE) |
| **A-E4-sim** (E4 이론 모의, CPU 8개) | 서버 CPU, tmux `A-E4-sim` | 07:25Z | 08:52Z | `~/pmx/A/logs/E4_sim.log` | 완료(오류 0) |
| **B-E4·E5** (M1·M3 기전과 처방) | 서버 GPU 1, tmux `B-E4-ftA`, `B-E45-laneB2`, `B-E45-laneC` | 03:35Z | 11:32Z | `~/pmx/B/logs/B-E4*.log`, `B-E45*.log` | 완료 |
| **A-E1-dinowm** (Wall·PointMaze, 280 job) | 서버 GPU 1, tmux `A-E1-dinowm` | 20:01Z | 13:15Z | `~/pmx/A/logs/E1_dinowm.log` | 완료 |
| B-E0-dp (PushT Diffusion Policy 3 seed, 1000 epoch) | 서버 GPU 1, tmux `B-E0-dp` | 13:32Z | 15:43Z | `~/pmx/B/logs/B-E0-dp*.log` | 완료 |
| A-E0-calib·held (TD-MPC2 보정·k 선정) | 서버 GPU 0 | 13:42Z | 16:38Z | `~/pmx/A/logs/E0_tdmpc2_*.log` | 완료 |
| b-e0 pod (Cosmos Policy·CureWM) | RunPod A40 | 약 13:20Z | 4시간 뒤 자동 종료 | pod `/workspace/logs/` | 완료(pod gone) |

## 사용자 확인이 필요한 사항

- **결정됨(사용자 승인, 2026-10-24 22:48Z)**: JKAIA 원고의 두 문장을 고쳤다(영문·국문 검토본·EasyChair 국문 초록, 두 PDF 모두 2쪽). `paper/README.md`에 기록.
- (위에서 결정) **(10-24 22:33Z 질문함) JKAIA 원고의 문장 두 곳**(마감 10-25 KST): 초록의 "overrates the plan it selects"는 DINO-WM PushT k=3·5에서 성립하지 않는다(det OEI 양수). 국문 검토본의 "두 계열의 손상 칸 어디에서도 det보다 나쁘지 않다"는 DMC walker k=3과 충돌한다. 추천안: 앞 구절은 빼고, 뒤 문장은 "Meta-World와 DINO-WM의 손상 칸"으로 고친다.
- **참고(10-24, 다른 세션)**: `A_dithered_world_models/paper/README.md`에 따르면 사용자가 JKAIA 원고의 주장 범위 "추천 1~5"를 승인했고, A 투고처를 JKAIA·TMLR·ICLR 2027 워크숍으로 바꿨다. 최종 논문(TMLR)용 주장 문구 13개는 FINAL_A.md §7에서 따로 정한다.
- **결정됨(사용자 승인, 2026-10-24 02:38Z)**: QAT는 약한 기준선이라는 단서를 달아 보고한다. DINO-WM 실험이 끝난 뒤 사후 점검으로 TD-MPC2 QAT+none 진단과 E7 지연 재측정만 돌린다(A DESIGN_main v5.3, 10절).
- (2026-10-23 22:30Z 질문함, 02:30Z 지연 문제와 합쳐 다시 물음, 위에서 결정) **약한 QAT와 E7 지연 잡음**
  - 사전등록한 TD-MPC2 QAT는 11칸 모두 QAT+det가 학습 없는 PTQ(B+det)보다 나빴다. QAT 보고 조건은 0/11로 걸리지 않았다.
  - E7 지연은 block 두 모드 때문에 회전별 오버헤드가 크게 흔들렸다. 사전 정의 계산으로는 H5 성립이다.
  - 추천안: 단서를 달아 보고하고, 사후 점검 두 가지(QAT+none, 지연 재측정)를 돌린다.
- **결정됨(사용자 승인, 2026-10-22 12:55Z)**: B H3은 사전등록대로 "성립(M2·M3)"으로 기록하고, 판정기 단서를 붙여 최종 보고에 반영한다(B DESIGN_main 11절).
- (2026-10-22 12:45Z 질문함, 위에서 결정) **B H3 판정의 누락 보충**
  - v2.2 §7이 정한 M2 H3 주 칸을 계산하지 않은 채로 남아 있었다. 계산하면 상상 몫이 0.829 [0.754, 0.886]이라, 사전등록 규칙상 H3은 M2·M3 두 모델에서 성립한다.
  - 추천안: 사전등록대로 H3 "성립(M2·M3)"으로 기록하고, M2 판정기의 영역 차이를 단서로 단다. 승인된 주장 범위(latent 모델 중심)는 그대로 두고, H3은 모델을 가로지르는 사전등록 결과로 함께 보고한다.
- **결정됨(사용자 승인, 2026-10-22 12:00Z)**: B 서술 범위 네 가지를 추천안대로 한다(B DESIGN_main v3.2).
- (2026-10-22 11:45Z 질문함, 위에서 결정) **B E4·E5 결과의 서술 범위**(B 실험 단계는 모두 끝남)
  - 추천안
    1. H4a: 통제 모델(M3)의 인과 결과로 쓴다. 성공 편중 자료가 near-miss 낙관을 만들고, 용량-반응이 단조롭다. 공개 DINO-WM(M1)의 짧은 미세조정에서는 성공 오판만 줄었으므로 부분 결과로 쓴다.
    2. H5a: 처방 주장에서 "경계 겨냥 반사실 자료"를 뺀다. 노출량을 맞추면 재가중보다 낫지 않다는 사전등록 부정 결과로 보고한다. 재가중 자체의 큰 효과(M3)는 탐색 결과로 표시한다.
    3. H5b: 규칙상 성립이지만 효과가 작다. "평가 쪽 할인이 near-miss 법칙과 FGRE를 줄인다"는 값싼 보정으로 쓴다.
    4. 예측기 용량 결과: "정보가 사라지는 위치" 절에 사전등록 밖 서술 결과로 표시해 넣는다.
- **결정됨(사용자 승인, 2026-10-22 09:50Z)**: A의 보완 추천안 1과 추천안 2를 그대로 진행한다(A DESIGN_main v5.2).
  - 기전 주장은 "반복에 걸쳐 지속되는 오차의 이용"으로 좁히고, OEI 진단 주장은 뺀다.
  - TD-MPC2 H2a 3칸은 독립 draw판으로 재측정한다(GPU 0 queue 첫 단계).
  - E2에 drift 가설을 E1 뒤 사전등록으로 더한다. DMC는 설계 7장 대응을 따른다.
- (2026-10-22 07:45Z 질문함, 위에서 결정) **A TD-MPC2 E1의 주장 수준 결과**
  - 기전(H2a)·진단(H2b) 불성립: sh-iter의 OEI도 유의하게 음수이고, OEI가 1·3-step 오차보다 손상을 더 잘 예측하지 못했다. 사후 분해로 보면 sh-iter의 음수는 마지막 평가 안의 선택 쏠림이고, 반복에 걸친 이동(drift)은 지속 오차 변형에서만 크다.
  - 추천안 1: H2a·H2b는 불성립 그대로 보고하고 분해는 사후로 표시한다. 기전 주장은 "반복에 걸쳐 지속되는 오차의 이용"으로 좁히고, "OEI가 1-step 오차보다 손상을 더 잘 예측한다"는 진단 주장은 뺀다(진단 기여는 H1과 계획 편차). E2에서는 drift 기반 가설을 E1 뒤 추가한 사전등록으로 명시하고 새 seed로 검정한다. 원래 H2c도 그대로 잰다.
  - 08:00Z 덧붙임: 두 계열의 OEI 구현이 다르다. DINO-WM은 실행 계획의 오차를 독립 draw로 다시 재고, TD-MPC2는 선택에 쓴 draw를 그대로 쓴다. 추천안 1에 더함: E2부터 두 계열 모두 두 판을 함께 재고, TD-MPC2 H2a 3칸은 독립 draw판을 더해 다시 돌린다(같은 seed, 사후 정정으로 표시).
  - DMC H3a 불성립(walker k=3에서 dither가 det보다 해로움). 추천안 2: 설계 7장 대응 그대로, E3에서 near-tie 밀도와 지속 편향 비중으로 설명을 시도한다. 설명이 안 되면 처방 주장을 Meta-World와 DINO-WM으로 좁히고 DMC는 해로운 칸을 포함해 한계로 보고한다.
- **결정됨(사용자 승인, 2026-10-21)**: B의 M2 판정기 규칙은 추천안대로 한다.
  - (c)는 끝 정렬 질의(마지막 k청크)로 쓰고, 문턱 0.5와 연속 V를 함께 보고한다.
  - (a)는 E0b에서 학습 데이터, 손목 영상, 끝 들어 올리기 동작을 더해 다시 시도한다. 실제 프레임 held-out 정확도가 0.95 이상이면 M2의 주 판정기로 쓴다. 미달이면 정확도를 밝힌 보조 판정기로 쓰고, 이때 M2의 주 판정기는 (c)다.
  - (d)는 들어 올리기 동작과 비교 프롬프트(7B), 그다음 32B VLM 순서로 시도한다. 안 되면 보조로 낮춘다.
  - 결과는 DESIGN_main.md에 고정한 뒤 E1을 시작한다.
- (2026-10-21 15:00Z 질문함, 위에서 결정) **B의 M2 판정기 사용 규칙**
  - 판정기 (d)(7B VLM)가 LIBERO 실제 프레임에서 우연 수준이다.
  - 판정기 (a)(oracle)가 0.88로 기준 0.95에 못 미친다.
  - 판정기 (c)(value head)는 남은 시간 할인이 판정을 지배한다.
  - 추천안: (c)는 끝 정렬 질의로 쓰고, (a)는 E1에서 다시 시도한다. (d)는 단계적으로 시도한 뒤 안 되면 보조로 낮춘다.
- **결정됨(사용자 승인, 2026-10-22 03:30Z)**: B는 규칙대로 주장을 M1·M3로 좁힌다. M2 E2는 생략하고, E4·E5만 서버에서 진행하며 E6는 생략한다(B DESIGN_main v3).
- (2026-10-22 01:05Z 질문함, 위에서 결정) **B의 go/no-go**
  - M2에서 H1a와 H1b가 모두 성립하지 않아, 사전등록 규칙상 B의 주장을 latent 월드모델(M1·M3)의 경계 해상도로 좁히고 워크숍이나 RA-L 경로로 바꿔야 한다.
  - 추천안: 규칙대로 좁힌다. M2 E1은 혼합 결과로 그대로 보고하고, M2 E2(약 $15)는 하지 않는다. M1·M3의 E2~E6은 서버에서 계속한다.
  - 03:05Z 갱신: 서버 E2·E3 결과, M1의 정책 수준 편향이 작아(H2a 불성립) 주장이 더 약해졌다. 추천을 바꿨다. 규칙대로 좁히고, B는 E4(특히 성공 편중 미세조정의 인과 시험)와 E5까지만 서버에서 돌린다. E6(checkpoint 선택)은 M1 편향이 작아 생략한다.
- **결정됨(사용자 승인, 2026-10-22)**: 세 가지 모두 추천안대로 한다.
  - PushT eval 쌍은 held와 프레임이 겹치지 않게 다시 뽑는다.
  - DINO-WM OEI 확증 지표는 비양자화 모델 비용 기준이다(참 비용은 보조). 설계 4.3절을 정정했다.
  - M3 n_π 주 구간은 [1, 1.2)다.
  - A DESIGN_main v3과 B DESIGN_main v2로 고정했다.
- (2026-10-21 20:05Z 질문함, 위에서 결정) **A DINO-WM 두 가지**
  - (1) **PushT eval 쌍 겹침.** 추천안: held 구간과 프레임이 겹치지 않게 eval 쌍을 다시 뽑는다. 파일럿 eval과의 겹침은 최소화하고 기록한다.
  - (2) **OEI 기준.** 추천안: 확증 지표는 비양자화 모델 비용 기준이다(파일럿 값의 정의, TD-MPC2와 같음, 양자화 오차만 분리). 시뮬레이터 참 비용 기준은 보조로 보고한다. 설계 문구는 정정한다.
- (2026-10-21 15:45Z 질문함) **B의 M3 near-miss 구간**
  - 설계 구간 [1, 2)로 재면 R2-9 장면의 far-failer(π5)와 near-failer(π6)의 대비가 사라진다.
  - 추천안: M3의 정책 법칙(H2a)에는 R2-9 구간 [1, 1.2)를 주 구간으로 사전등록하고, [1, 2)는 함께 보고한다.
- 조정자가 설계 범위 안에서 정한 것(B 서버 E0 뒤, 사용자 보고만)
  - M1의 정책 법칙 주 분석은 DP checkpoint 24개로 하고, R2-12 교란 정책 8개는 통제 대비로 따로 본다. 검정력은 0.95에서 0.89로 내려가며, 설계 6.7절 문구를 고친다.
  - 표준 PushT(DP)에는 agent 목표가 없다. 그래서 DP에는 물체 자세 r_obj와 표준 목표 판별기 a_std(held-out 0.975)를 쓰고, 원 정의 r과 판정기 (b)는 교란 정책 통제에만 쓴다.
  - 판별기 (a)는 5-seed 앙상블로 한다. 0.5 문턱 운영점이 seed마다 흔들렸기 때문이다.
  - 교란 정책 통제의 보조 평가 집합으로 R2-12의 trainsplit을 둔다(val 궤적이 21개뿐이라서).
  - DP 단계 8개(epoch 20·25·30·40·50·80·220·1000)를 확정한다.
  - GPU 재배정: B 서버 E0이 끝나 GPU 1이 비었다. B의 E1이 시작될 때까지 A의 DINO-WM이 GPU 1을 쓴다.
- 조정자가 설계 범위 안에서 정한 것(사용자 보고만)
  - M2 과제 6개(t1, t2, t4, t5, t6, t8)
  - 경계 겨냥 교란 계열 추가(설계의 "경계 근처 촘촘한 severity grid" 실현)
  - M2 전체 A40 고정(재고가 없으면 전체를 한 종류로 이동)
  - LIBERO DP 학습 상한 `b-dp` $5 추가

## 점검 기록

| 시각(UTC) | 결과 |
|---|---|
| 16:39 (10/24) | **A의 단계 분석이 모두 끝났다**(E0 ~ E7, 사후 점검). 서버에서 도는 작업 없음. 남은 것은 A 최종 보고와 주장 문구 결정이다. 정기 점검은 멈춘 그대로다 |
| 16:28 (10/24, 사후 queue 끝 감지) | **사후 점검 완료**(POSTHOC_DONE 16:26Z). **A의 서버 GPU 작업이 모두 끝났다**: tmux 없음, MPS 꺼짐, 두 GPU 빔. 남은 일은 분석뿐이다: DINO-WM E2 EO(기술)와 사후 점검 정리(16:30Z에 맡김), 그다음 A 최종 보고. 정기 점검은 멈춘 그대로다 |
| 14:59 (10/24, 사용자 요청으로 한 번 점검, 정기 점검은 멈춘 그대로) | **A 본 실험 실행이 모두 끝났다**: DINO-WM E2 EO 150/150, GPU 0 사슬 14:16Z·GPU 1 사슬 14:41Z ALL_DONE, 실패 claim 없음. **사후 점검 1(QAT+none) 완료**(14:46Z, 198 job): QAT 학습 자체가 모델을 크게 나쁘게 만들었다(11칸 모두 +0.20~+0.82). **사후 점검 2(지연 재측정) 진행**(14:51Z 시작, GPU 0 배타, 약 16:25Z 끝). GPU 1은 비었다. 디스크 여유 264GB. RunPod pod 없음, 누적 $20.27, 감시 스크립트 정상 |
| 11:56 (10/24, 사용자 요청으로 한 번 점검, 정기 점검은 멈춘 그대로) | 서버 GPU 0·1 모두 100%, tmux 3개 정상, 디스크 여유 264GB. **DINO-WM E5 전부 완료**: 가중치 60/60, QAT 학습과 QAT 평가 30/30(11:51Z, GPU 1). E2 EO 52/150: top6 완료(GPU 0, 09:47Z), top38 22/30(GPU 0), top75 시작(GPU 1, 11:51Z). it10·it5 칸이 남았다. 실제 traceback 0, 실패 claim 없음. A 실험은 약 14:30Z, 사후 점검은 약 16:30Z에 끝날 것으로 본다. 사후 queue는 대기 중(정상). RunPod pod 없음, 누적 $20.27, 감시 스크립트 정상 |
| 06:35 (10/24, 사용자 요청으로 한 번 점검, 정기 점검은 멈춘 그대로) | 서버 GPU 0·1 모두 100%, tmux 3개(DINO-WM 사슬 둘, 사후 queue) 정상, 디스크 여유 264GB. **DINO-WM E5 평가 완료**(60/60: GPU 0이 PushT k=4, GPU 1이 PushT k=3·PointMaze k=3). E5 가중치 48/60(GPU 0 w4sr4, GPU 1 w8rtn 진행). 실제 traceback 0, 실패 claim 없음. 남은 순서: GPU 1은 QAT 학습(최대 2시간) → E5 QAT 평가 → E2 EO, GPU 0은 E2 EO. 지금 속도면 A 실험은 약 15Z, 사후 점검(QAT+none, 지연)은 약 16:30Z에 끝난다. 사후 queue는 대기 중(정상). RunPod pod 없음, 누적 $20.27, 감시 스크립트 정상 |
| 02:54 (10/24) | **사용자 요청으로 정기 점검을 멈춘다(다시 시작하라고 할 때까지).** 조정자의 1시간 타이머를 멈췄고, 작업자 감시는 없다. 서버 작업은 그대로 돈다: GPU 0 DINO-WM 사슬(`A-v5-dinowm-g0`, E5 평가 PushT k=4 칸부터), GPU 1 DINO-WM 사슬(`A-v5-dinowm-g1`, E5 평가 → QAT → E2 EO), 사후 점검 queue(`A-posthoc-g0`, GPU 0 사슬이 끝나면 자동 시작). A 실험은 10-24 16~18Z, 사후 점검은 그 뒤 약 2시간 안에 끝날 것으로 본다. 실패한 DINO-WM 칸은 5번 시도 뒤 `failed`로 표시되고 사슬은 다음으로 넘어간다. 점검을 다시 시작하면 `runpod_main/check_all.sh`와 `runpod_main/check_a_detail.sh`(A 세부, 새로 둠)로 확인한다. RunPod pod 없음, 누적 $20.27, 예산 감시 스크립트는 켜 둔다 |
| 02:19 | 서버 GPU 0·1 모두 100%, tmux 2개(`A-v5-dinowm-g0`, `A-v5-dinowm-g1`) 정상, 디스크 여유 265GB. GPU 0 사슬이 E5_dinowm_eval의 PushT k=4 칸을 잡아 돈다(episode당 33.8초). GPU 1은 PushT k=3 칸 7/20, traceback 0. QAT 단계는 GPU 1에 맡겨, GPU 0이 QAT 학습(약 2시간)을 기다리지 않고 E2 EO로 가게 `plans/v5_g0.txt` 순서를 조정했다(실행 순서만). RunPod pod 없음, 누적 $20.27, 감시 스크립트 정상 |
| 02:17 (GPU 0 queue 끝) | **TD-MPC2 queue 완료**(QUEUE_DONE 02:15Z): E7 성공률 1,296/1,296(01:29Z, 실패 0), MPS 멈춤, E7 지연 5회전(01:29~02:15Z, `e7_latency.json`). GPU 0이 비어 **DINO-WM 두 번째 사슬을 GPU 0에 띄웠다**(02:17Z, tmux `A-v5-dinowm-g0`, `plans/v5_g0.txt`, v5 §7.5 계획). GPU 1은 E5 평가 PushT k=3 칸 진행(6/20). E7 분석을 TD-MPC2 작업자에게 맡겼다 |
| 01:17 | 서버 GPU 0·1 모두 100%, tmux 2개 정상, 디스크 여유 265GB. GPU 0: TD-MPC2 E7 성공률 1,159/1,296 job, 실패 0(약 01:42Z 끝, 이어 MPS 멈춤과 E7 지연, queue 끝은 약 02:35Z). GPU 1: DINO-WM E5 held 34/42 job. PushT k=4 칸이 01:02Z에 끝났고 PointMaze k=3 칸 6/14 진행(held 끝 약 01:35Z), traceback 0. GPU 0 queue가 끝나면 GPU 0에도 DINO-WM 사슬을 띄울 준비를 했다(`plans/v5_g0.txt`, v5_g1.txt와 같은 단계, 칸은 claim으로 나눔). RunPod pod 없음, 누적 $20.27, 감시 스크립트 정상 |
| 00:16 (10/24) | 서버 GPU 0·1 모두 100%, tmux 2개 정상, 디스크 여유 265GB. GPU 0: TD-MPC2 E7 성공률 749/1,296 job, 실패 0(약 01:45Z 끝). GPU 1: DINO-WM E5 held 19/42 job. PushT k=3 칸이 23:42Z에 끝났고 PushT k=4 칸 5/14 진행, traceback 0(held 끝은 약 02Z). RunPod pod 없음, 누적 $20.27, 감시 스크립트 정상 |
| 23:15 | 서버 GPU 0·1 모두 100%, tmux 2개 정상, 디스크 여유 265GB. GPU 0: TD-MPC2 E7 성공률 391/1,296 job, 실패 0(약 01:45Z 끝, 이어 MPS를 멈추고 E7 지연 측정, 약 02:45Z까지). GPU 1: DINO-WM E5 held 8/42 job(PushT k=3 칸 8/14), traceback 0. held 뒤 `e5_select`가 E5 평가 manifest를 쓰는 것을 코드로 확인했다(사슬이 멈추지 않음). E2·E3·E5 분석 네 건은 모두 끝났다. RunPod pod 없음, 누적 $20.27, 감시 스크립트 정상 |
| 22:13 | 서버 GPU 0·1 모두 100%, tmux 2개 정상, 디스크 여유 265GB. **TD-MPC2 E5 완료**(평가 1,728/1,728, 22:11Z, 실패 0). E7 성공률(1,296 job)이 22:11Z에 시작했고, 단계 전환에서 MPS는 정상이었다. E5 분석을 TD-MPC2 작업자에게 맡겼다. GPU 1: DINO-WM E3 124/130 job(Wall k=3 칸 14/20), traceback 0. 약 22:25Z에 끝나면 E3 분석을 DINO-WM 작업자에게 맡긴다. RunPod pod 없음, 누적 $20.27, 감시 스크립트 정상 |
| 21:12 | 서버 GPU 0·1 모두 100%, tmux 2개 정상, 디스크 여유 265GB. GPU 0: TD-MPC2 E5 평가 1,264/1,728 job, 실패 0(약 22:30Z 끝 예상). GPU 1: DINO-WM E3 91/130 job. PushT k=3 칸이 21:07Z에 끝났고 Wall k=5 칸 진행(episode당 5.4초), traceback 0. Wall k=3까지 E3 끝은 약 21:50Z. RunPod pod 없음, 누적 $20.27, 감시 스크립트 정상 |
| 20:11 | 서버 GPU 0·1 모두 100%, tmux 2개 정상, 디스크 여유 265GB. GPU 0: TD-MPC2 E5 평가 900/1,728 job, 실패 0(약 22:10Z 끝 예상). GPU 1: DINO-WM E3 82/130 job. PointMaze k=3 칸이 19:57Z에 끝났고 PushT k=3 칸 2/10 진행, traceback 0. 남은 Wall 칸까지 E3 끝은 약 21:35Z. RunPod pod 없음, 누적 $20.27, 감시 스크립트 정상 |
| 19:10 | 서버 GPU 0·1 모두 100%, tmux 2개 정상, 디스크 여유 265GB. GPU 0: TD-MPC2 E5 평가 471/1,728 job, 실패 0(약 22:40Z 끝 예상). GPU 1: DINO-WM E3 61/130 job. PushT k=4 칸이 18:12Z에 끝났고 PointMaze k=3 칸 21/40 진행(episode당 15초), traceback 0. E3 끝은 약 21:30Z. RunPod pod 없음, 누적 $20.27, 감시 스크립트 정상 |
| 18:09 (단계 전환 확인) | GPU 0 queue가 MPS 문제 없이 두 번 넘어갔다. **E5 held 완료**(17:47Z, 1,188/1,188, 실패 0), **QAT 학습 완료**(17:52Z, 33/33, 약 5분), 기준선 선택(`selection.json`)도 끝났다. 이어 E5 평가가 17:52Z에 시작했다(1,728 job 중 60, 실패 0). 지금 속도면 E5 평가는 약 22Z, E7 성공률은 약 01Z, E7 지연까지 약 02~03Z(10/24)에 끝난다. GPU 1: DINO-WM E3 38/130(PushT k=4 칸 38/40), traceback 0. 디스크 여유 265GB |
| 17:31 | 서버 GPU 0·1 모두 100%, tmux 2개 정상, 디스크 여유 265GB. GPU 0: TD-MPC2 E5 held 1,031/1,188 job, 실패 0(약 17:55Z 끝 예상). 다음 단계(QAT 학습)로 넘어갈 때 MPS 고장이 다시 생기는지 보려고 다음 점검을 18:08Z 무렵으로 당긴다. GPU 1: DINO-WM E3 32/130 job(PushT k=4 칸 32/40), traceback 0. RunPod pod 없음, 누적 $20.27, 감시 스크립트 정상 |
| 16:30 | 서버 GPU 0·1 모두 100%, tmux 2개 정상, 디스크 여유 266GB. GPU 0: TD-MPC2 E5 held 578/1,188 job, 실패 0(MPS client 16, 약 18:15Z 끝 예상). GPU 1: DINO-WM E3 23/130 job(PushT k=4 칸 23/40), traceback 0, 실패 claim 없음(E3 끝은 약 22Z). E2·E3 분석 두 건 완료(TD-MPC2 16:05Z, DINO-WM N sweep 15:55Z). RunPod pod 없음, 누적 $20.27, 감시 스크립트 정상 |
| 15:27 (10/23, 사용자 요청으로 정기 점검 재개) | 맥북이 재부팅돼(약 15:14Z) Tailscale과 RunPod 예산 감시 스크립트가 꺼져 있었다. 사용자가 Tailscale을 켰고, 감시 스크립트는 다시 켰다(pod 없음, 누적 $20.27). 서버 GPU 0·1 모두 100%, tmux 2개 정상, 디스크 여유 266GB. 08:35Z 점검 뒤 실패는 0이다. **TD-MPC2 E3 완료**(14:51Z, 2,124/2,124, 실패 0, 6.3시간). E3 → E5 전환에서 MPS는 정상이었다. E5 통계는 10-22 준비 때 이미 끝나 있었고, E5 held가 212/1,188 진행 중이다(MPS client 16). **DINO-WM E2 N sweep 완료**(13:55Z, 150/150). 지금 E3 13/130(PushT k=4, episode당 37초)이고 실패 claim은 없다. 로그의 traceback 줄은 모두 multiprocessing 정리 메시지다 |
| 08:35 (10/23, 사용자 요청으로 한 번 점검) | 맥북 Tailscale이 꺼져 있어 처음에는 서버에 접속하지 못했다(사용자가 다시 켬). **TD-MPC2 E2 완료**(06:18Z, 4,482/4,482, 실패 0). **오류 발견·수정**: 그 뒤 GPU 0 queue가 E3 시작 전 점검에서 멈췄다. MPS server가 새 client를 받지 못했다(Error 205, 외부 프로세스 없음, /dev/shm 정상). GPU 0은 약 2.3시간 놀았다. MPS daemon을 다시 켜 시험 client가 정상으로 붙는 것을 확인했다. DINO-WM: E2 N1024 칸 완료(04:50Z), N512 칸 진행. RunPod 예산 감시 스크립트가 꺼져 있어 다시 켰다(pod 없음, $20.27). 디스크 여유 266GB |
| 02:15 | **사용자 요청으로 정기 점검을 잠시 멈춘다.** 조정자의 1시간 타이머와 두 작업자의 감시를 멈췄다. 서버 작업(GPU 0 TD-MPC2 queue E2 진행, GPU 1 DINO-WM 사슬 E2 진행)은 그대로 돈다. queue는 job마다 결과 파일을 쓰고, 실패하면 한 번 다시 시도한 뒤 `.failed`로 표시하고 다음 단계로 넘어간다. 그래서 점검을 다시 시작하면 그때 실패 기록을 확인해 다시 돌리면 된다 |
| 02:11 | 서버 GPU 0·1 모두 100%, 디스크 여유 266GB. GPU 0: TD-MPC2 E2 2,512/4,482 job, 실패 0(지난 1시간 약 220 job, 무거운 job 구간). GPU 1: DINO-WM E2 N1024 칸 210/300 run. RunPod pod 없음, 누적 $20.27, 감시 스크립트 정상 |
| 01:10 | 서버 GPU 0·1 모두 100%, 디스크 여유 266GB. GPU 0: TD-MPC2 E2 2,292/4,482 job, 실패 0(평균 시간당 약 410, 약 06:30Z 끝 예상). GPU 1: DINO-WM E2 N1024 칸 180/300 run(약 05Z 끝). RunPod pod 없음, 누적 $20.27, 감시 스크립트 정상 |
| 00:09 (10/23) | 서버 GPU 0·1 모두 100%, 디스크 여유 266GB. GPU 0: TD-MPC2 E2 1,707/4,482 job, 실패 0(지금은 무거운 cheetah 반복 12회 job이라 지난 1시간 약 220 job). GPU 1: DINO-WM E2 N1024 칸 150/300 run(약 05Z 끝). RunPod pod 없음, 누적 $20.27, 감시 스크립트 정상 |
| 23:08 | 서버 GPU 0·1 모두 100%, 디스크 여유 266GB. GPU 0: TD-MPC2 E2 1,489/4,482 job, 실패 0(지난 1시간 약 585 job, 약 05Z 끝 예상). GPU 1: DINO-WM E2 N1024 칸 120/300 run. RunPod pod 없음, 누적 $20.27, 감시 스크립트 정상 |
| 22:08 | 서버 GPU 0·1 모두 100%, 디스크 여유 267GB. GPU 0: TD-MPC2 E2 904/4,482 job, 실패 0(시간당 약 360). GPU 1: DINO-WM E2 N1024 칸 90/300 run. RunPod pod 없음, 누적 $20.27, 감시 스크립트 정상 |
| 21:07 | 서버 GPU 0·1 모두 100%, 디스크 여유 267GB. GPU 0: TD-MPC2 E2 545/4,482 job, 오류 0(1.6시간 동안 시간당 약 350 job, 약 06~07Z 끝 예상). GPU 1: DINO-WM E2 N1024 칸 60/300 run(episode당 128초, 이 칸만 약 05Z까지). RunPod pod 없음, 누적 $20.27, 감시 스크립트 정상 |
| 20:05 | 서버 GPU 0·1 모두 100%, 디스크 여유 267GB. GPU 0: H2a 재측정 완료(19:33Z, 162/162, E1과 비트 단위로 같음). 이어 E2 108/4,482 job, 오류 0(가장 무거운 cheetah N1024 job부터 돌아 아직 속도 추정은 이르다). GPU 1: DINO-WM E2 N1024 칸, episode당 128초(예상대로). A go/no-go = go 기록. RunPod pod 없음, 누적 $20.27, 감시 스크립트 정상 |
| 19:14 | **A E1 실행 완료**: PushT k=5(GPU 1) 18:55Z, k=3(GPU 0) 18:56Z. **v5 사슬 둘이 자동으로 시작했다.** GPU 0 queue는 19:05Z에 MPS를 켜고 H2a 재측정을 돌린다(51/162, E1과 비트 단위로 같음). GPU 1 사슬은 E2 N sweep(N1024 칸)부터 돈다. 처음에는 v5.2 영역 효과 계산(PushT k=3·5, 방금 끝남)과 겹쳐 느렸다. 디스크 여유 267GB. 다음은 DINO-WM E1 분석과 A go/no-go다. RunPod pod 없음, 누적 $20.27, 감시 스크립트 정상 |
| 18:34 | 서버 GPU 0·1 모두 100%, 디스크 여유 267GB. PushT: k=3 460/500(GPU 0), k=5 460/500(GPU 1). 약 19:00Z에 E1 전체가 끝날 예정이다. 그 뒤 GPU 0 queue(TD-MPC2, MPS)와 GPU 1 사슬(DINO-WM)이 자동으로 시작한다. 시작 확인은 19:15Z 무렵에 한다. 오류 없음. RunPod pod 없음, 누적 $20.27, 감시 스크립트 정상 |
| 17:34 | 서버 GPU 0·1 모두 100%, 디스크 여유 268GB. PushT: k=3 360/500(GPU 0), k=5 360/500(GPU 1). 둘 다 약 19:00Z에 끝 예상. 오류 없음. v5 사슬 둘 대기. RunPod pod 없음, 누적 $20.27, 감시 스크립트 정상 |
| 16:33 | 서버 GPU 0·1 모두 100%, 디스크 여유 268GB. PushT: GPU 0 k=3 260/500, GPU 1 k=5 260/500(episode당 37초). 둘 다 약 19:00~19:05Z에 끝 예상. 오류 없음. v5 사슬 둘 대기. RunPod pod 없음, 누적 $20.27, 감시 스크립트 정상 |
| 15:32 | 서버 GPU 0·1 모두 100%, 디스크 여유 268GB. PushT: GPU 0 k=3 160/500, GPU 1 k=5 170/500(둘 다 episode당 37초, 약 19:00~19:05Z에 끝). v5.2 영역 효과 계산(PointMaze)은 끝났다(15:00Z 무렵, 점검 일치). v5 사슬 둘은 대기 중. RunPod pod 없음, 누적 $20.27, 감시 스크립트 정상 |
| 14:32 | 서버 GPU 0·1 모두 100%, 디스크 여유 268GB. PushT: GPU 0은 k=4를 마치고(13:52Z) k=3 60/500(episode당 37초, 약 19:05Z 끝)이다. GPU 1의 k=5는 90/500이다. DINO-WM 작업자의 v5.2 영역 효과 계산(`A-v52-g1`, 약 1시간)과 GPU 1을 나눠 episode당 75초로 느려졌다(끝 예상 약 19:20Z, go/no-go 시점은 거의 같음). v5.2 PushT k=4 중간 결과: H2a(=H2a_indep)와 H2a′ 모두 PushT에서 성립, det drift의 약 2/3는 영역 효과. v5 사슬 둘 대기. RunPod pod 없음, 누적 $20.27, 감시 스크립트 정상 |
| 13:31 | 서버 GPU 0·1 모두 100%, 디스크 여유 268GB. **DINO-WM Wall·PointMaze E1 완료**(13:15Z, 280 job, 누락 0). PushT: GPU 0의 k=4 칸 660/700(약 13:55Z 끝, 이어 k=3은 약 19:05Z), GPU 1이 13:16Z부터 k=5를 시작(episode당 37초, 약 18:30Z 끝). v5 사슬 둘은 대기 중. B 최종 보고 완료(13:05Z). RunPod pod 없음, 누적 $20.27, 감시 스크립트 정상 |
| 12:30 | 서버 GPU 0·1 모두 100%, 디스크 여유 268GB. B의 서버 작업과 tmux는 모두 끝났고 정리도 끝났다(B 15GB). DINO-WM: PointMaze가 GPU 1 단독으로 episode당 15초, 1,510/1,700(약 13:20Z에 끝 예상). PushT는 GPU 0에서 550/1,700(k=4 칸 550/700). 예상: k=4 약 14:05Z, k=3 약 19:15Z, GPU 1의 k=5(PointMaze 뒤)는 약 18:30Z. v5 사슬 둘은 대기 중. RunPod pod 없음, 누적 $20.27, 감시 스크립트 정상 |
| 11:28 | 서버 GPU 0·1 모두 100%, 디스크 여유 266GB(B가 bank를 정리해 18GB). DINO-WM: PointMaze 1,270/1,700(B lane 하나가 끝나 episode당 30~33초), PushT 450/1,700(k=4 칸 450/700, episode당 37초). 10:27 기록의 "k=4 약 11Z, k=3 약 16Z"는 잘못 센 것이다. 바른 예상은 k=4 약 14:15Z, k=3 약 19:30Z다. v5 사슬 둘은 대기 중(queue 시간별 줄 정상). B: lane B2가 끝났고(11:08Z) lane C의 재상상이 남았다. RunPod pod 없음, 누적 $20.27, 감시 스크립트 정상 |
| 10:27 | 서버 GPU 0·1 모두 100%, 디스크 여유 231GB(B 52GB로 정리됨, A 4.2GB). DINO-WM PushT: k=4 칸의 마지막 변형(det+shift) 진행, GPU 0 단독 episode당 37초. k=4는 약 11Z, k=3은 약 16Z에 끝날 것으로 본다. PointMaze: B와 GPU 1을 나눠 episode당 45초. v5 사슬 둘은 무장 상태로 대기하고, H2a 재측정 flag를 확인했다. B: 재상상 lane과 설명용 sweep을 돌리고, bank를 정리 중(B GPU 작업은 약 11:40Z에 끝). RunPod pod 없음, 누적 $20.27, 감시 스크립트 정상 |
| 09:27 | 서버 GPU 0·1 모두 100%. **디스크 여유 223GB(1시간에 34GB 감소)**: B가 26 → 60GB(feat 49GB, E4·E5 특징 bank)로 늘었다. B 작업자에게 최대 크기, 정리 계획, 끝나는 시각을 물었다(기준: 여유 60GB 아래면 멈춤, B 최대 80GB 목표). DINO-WM PushT: GPU 0 단독 episode당 37초(k=4 칸 ind 변형 진행). PointMaze: 1,050/1,700 이상. v5 사슬 둘은 무장 상태로 대기한다. B: E4 미세조정 lane이 끝났고(09:22Z), E4·E5 평가 lane 2개가 남았다. A E4 완료. RunPod pod 없음, 누적 $20.27, 감시 스크립트 정상 |
| 08:26 | 서버 GPU 0·1 모두 100%, 디스크 여유 257GB. DINO-WM PushT: GPU 0 단독으로 episode당 30.5초(none 3개 job과 nonealt 진행). PointMaze: B와 GPU 1을 나눠 episode당 46초. A E4 모의 본 실행 진행(CPU 8개). A v5 점검(동일성 재실행, E5 스모크)은 모두 통과했고 지금 GPU 0에는 PushT만 있다. B: E4 미세조정 진행(cp 조건). RunPod pod 없음, 누적 $20.27, 감시 스크립트 정상 |
| 07:26 | 서버 GPU 0·1 모두 100%, 디스크 여유 258GB. A TD-MPC2: E1 분석 완료(사용자 결정 대기), v5 초안용 스모크 27/27 통과, MPS 없는 처리량 측정 중(약 10분, 작업자 2개). DINO-WM: PointMaze가 B와 GPU 1을 나눠 episode당 38초, PushT k=4 칸 진행 중(GPU 0을 처리량 측정과 잠시 나눠 episode당 약 65초). A E4 모의: 스모크 71 job을 마치고 본 실행을 시작했다(CPU 8개). B: E4 미세조정과 E5 lane 진행. 로그의 error 비슷한 줄 18개는 이전에 확인한 multiprocessing 정리 메시지다. RunPod pod 없음, 누적 $20.27, 감시 스크립트 정상 |
| 06:19 | **오류 발견·수정**: TD-MPC2 E1의 episode 4,104개는 06:08Z에 모두 끝났다. 그런데 이어진 칸 단위 job 95개(prederr 83, decision 12)가 모두 CUDA 초기화에서 실패했다("Error 205"). 같은 시각 PushT E1이 MPS 없이 GPU 0에서 시작한 뒤 새 MPS client가 붙지 못한 것으로 보인다. A의 MPS daemon을 멈추고, 칸 단위 job을 MPS 없이 작업자 2개로 다시 띄웠다(06:20Z, `A-E1-tdmpc2-cells`). 06:24Z까지 16개 성공, 실패 0. 결과 손실은 없다. DINO-WM PointMaze: 193/280(GPU 1을 B와 공유해 episode당 47초). PushT: GPU 0에서 첫 job 완료. B: E4 미세조정(S50 seed 2)과 E5 MC dropout 진행 중. 디스크 여유 259GB. RunPod pod 없음, 누적 $20.27, 감시 스크립트 정상 |
| 05:19 | 서버 GPU 0·1 모두 100%, 디스크 여유 259GB. TD-MPC2 E1: 3,507/4,009(episode 단계는 08:15Z 무렵 종료 예상). DINO-WM PointMaze: 183/280. B와 GPU 1을 나눠 episode당 46초다. B가 12:00Z 무렵 끝나면 빨라져 13:30Z 무렵 완료 예상이다. 예상 일정: PushT는 08:15Z부터 GPU 0, 13:30Z부터 GPU 1에서 돌아 A E1 전체가 20:00Z 무렵 끝난다. B: S97 미세조정 seed 2개를 마쳤고 M3 E5 재학습이 진행 중이다. RunPod pod 없음, 누적 $20.27, 감시 스크립트 정상 |
| 04:18 | 서버 GPU 0·1 모두 100%, 디스크 여유 260GB. TD-MPC2 E1: 이번 실행에서 3,332/4,009 job. 남은 것이 무거운 DMC라 시간당 약 170개로, episode 단계는 08:20Z 무렵 끝날 예정이다. DINO-WM PointMaze: 174/280. B의 E4 미세조정과 E4·E5 lane 2개가 GPU 1을 같이 써서 episode당 40초 안팎으로 느려졌다(사용자에게 알린 A 지연). B: E4 미세조정 자료 생성과 E4c(M1)를 마쳤고, S97 미세조정을 진행 중이다. RunPod pod 없음, 누적 $20.27, 감시 스크립트 정상 |
| 03:18 | 서버 GPU 0·1 모두 100%, 디스크 여유 269GB. TD-MPC2 E1: episode 3,718/4,104. 남은 것이 무거운 DMC job이라 시간당 약 170개로 느려졌고, 05:30Z 무렵 episode 단계가 끝날 예정이다. DINO-WM PointMaze: 152/280. B 작업이 끝나 GPU 1 단독이 됐고 episode당 15초다(08:40Z 무렵 완료 예상). PushT E1은 두 GPU의 시작 조건을 기다린다. RunPod pod 없음, 누적 $20.27, 감시 스크립트 정상 |
| 02:17 | 서버 GPU 0·1 모두 100%, 디스크 여유 269GB. TD-MPC2 E1: episode 3,544/4,104(시간당 약 530, 03:30Z 무렵 episode 단계 종료 예상). 그 뒤 GPU 0에서 PushT E1이 자동 시작한다. DINO-WM PointMaze: 133/280(GPU 1을 B VLM과 공유, episode당 23초). B VLM (d): DP H5 상상 프레임 판정 중. RunPod pod 없음, 누적 $20.27, 감시 스크립트 정상 |
| 01:17 | 서버 GPU 0·1 모두 100%, 디스크 여유 269GB. TD-MPC2 E1: episode 3,018/4,104. DINO-WM: Wall 전부 완료(00:52Z), PointMaze 진행(116/280). 로그의 traceback 18줄은 프로세스 종료 때 나는 multiprocessing 세마포어 정리 경고였다(결과 누락 0). B 서버: E2 M1 통제 집합 분석과 E3 1차 분석(H3: M1 불성립, M3 성립)을 마쳤고 VLM (d) 규칙 집합을 판정 중이다. RunPod pod 없음, 누적 $20.27, 감시 스크립트 정상 |
| 00:16 (10/22) | 서버 GPU 0·1 모두 100%, 디스크 여유 270GB. TD-MPC2 E1: episode 2,787/4,104(시간당 약 470, 03:00Z 무렵 episode 단계 종료 예상). DINO-WM Wall·PointMaze: 97/280. B E2·E3 서버: DP 24개 rollout과 통제 집합 완료, E1 곡선 재생성 일치(차이 5e-7), Qwen2.5-VL-7B 설치 뒤 (d) 판정 중. RunPod: b-e1 pod 8대, 작업자 기록상 오류 0이고 00:40Z 무렵 끝날 예정이다. b-e1 약 $11.8 / 상한 $18. 누적 $18.52, 감시 스크립트 정상 |
| 23:15 | 서버 GPU 0·1 모두 100%, 디스크 여유 283GB. TD-MPC2 E1: episode 2,316/4,104. DINO-WM Wall·PointMaze: 71/280. GPU 1을 B의 M1 E1 lane 3개와 나눠 Wall이 episode당 17초다. PushT E1은 새 eval 쌍과 manifest가 준비됐고, GPU 0 쪽은 TD-MPC2 episode 단계 종료를, GPU 1 쪽은 Wall·PointMaze 종료를 기다린다. B: 판정기 a_std 앙상블 0.991, (b) 고정 문턱 held-out 0.887·0.897, M1 곡선 lane 진행 중. RunPod: b-e1 pod 8대(A40)가 M2 E1을 돌린다(b-e1 $6.9 / 상한 $18, 시간당 약 $4.7). 누적 $13.60, 감시 스크립트 정상 |
| 22:14 | 서버 GPU 0·1 모두 100%, 디스크 여유 283GB. TD-MPC2 E1: episode 2,018/4,104(시간당 약 400). DINO-WM E1: 55/280(GPU 1을 B와 공유해 느림). B 서버: M3 E1을 마치고 판정 계산까지 했다. 판정기 a_obj 앙상블은 held-out 0.968로 통과했고 M1 곡선 작업 중이다. RunPod: b-e1 pod 6대(A40)가 M2 E1을 나눠 돌린다. 누적 $8.62(b-e1 $1.82, 상한 $18), 감시 스크립트 정상 |
| 21:14 | 서버 GPU 0·1 모두 100%, 디스크 여유 293GB. TD-MPC2 E1: episode job 1,614/4,104(시간당 약 360), 오류 0. DINO-WM E1: 41/280. B의 E1 보정과 M3 학습이 GPU 1을 같이 써서 Wall이 episode당 11초로 느려졌다(예상한 공유 비용). B M2: 아직 pod 없음(준비 중). RunPod 누적 $6.70, 감시 스크립트 정상 |
| 20:14 | 서버 GPU 0·1 모두 100%, 디스크 여유 293GB. TD-MPC2 E1: episode job 1,254/4,104 완료, 재시작 뒤 오류 0. DINO-WM E1(Wall·PointMaze): 6/280 완료, Wall 평가 중(episode당 4.6초). RunPod: b-dp pod만 실행 중이다(b-dp 누적 $3.15, 상한 $5). 상한에 가까워지면 E0b 작업자가 정리하는지 확인한다. 전체 누적 $6.20, 감시 스크립트 정상 |
| 19:10 | **오류 발견·수정**: TD-MPC2 E1에서 칸 단위 job(prederr·decision) 95개가 GPU 메모리 부족으로 실패했다(작업자 16개 동시 실행). episode job 911개는 정상이었다. 실행 순서만 바꿔 다시 띄웠다(19:11Z, `run_e1_split.sh`): episode는 16개씩, 칸 단위 job은 그 뒤 3개씩 돌린다. 끝난 job은 건너뛰므로 잃은 결과는 없다. DINO-WM은 GPU 1에서 PushT 재현 중이다. RunPod: b-e0 pod이 끝나 지워졌고(E0b 판정기 작업 종료), b-dp가 실행 중이다. 누적 $5.55, 감시 스크립트 정상 |
| 18:09 | A TD-MPC2: 처리량 측정을 마쳤다. Meta-World는 MPS 작업자 16개로 GPU 1장당 시간당 약 4,800 episode다(MPS 없이는 약 2,100). GPU 0이 비어 DINO-WM 작업자에게 GPU 두 장을 다 쓰라고 알렸다. DINO-WM은 PushT 재현을 진행 중이다(GPU 1, episode당 37초). RunPod: b-dp는 DP 초기 checkpoint 평가를 마쳤다. b-e0은 판정기 (a) v2(학습 5,080장, 로지스틱과 CNN)를 학습 중이다. 누적 $4.65, 감시 스크립트 정상 |
| 17:09 | 서버 GPU 0·1 모두 100%, 디스크 여유 293GB. A: TD-MPC2 held 보정과 k 선정이 16:38Z에 끝났고 지금은 MPS 작업자 수별 처리량을 잰다. DINO-WM은 GPU 1 단독에서 PushT가 episode당 37초다(4090 파일럿의 약 2배). E1 계산량은 E0 보고를 받아 다시 정한다. RunPod: b-e0에서 E0b 검증 작업 1056/1056과 판정기 스모크 8/8을 오류 없이 마쳤고 32B VLM 판정을 실행했다. b-dp는 DP 초기 checkpoint를 평가 중이다. 누적 $3.43, 감시 스크립트 정상 |
| 16:08 | 서버 GPU 0·1 모두 100%, 디스크 여유 293GB. B의 DP 학습이 15:43Z에 끝났다(3 seed 모두 정상 종료, checkpoint 8.9GB). A: TD-MPC2 held 평가가 600개 중 278개를 마쳤다(18:00Z 무렵 완료 예상). DINO-WM의 PushT는 GPU 0에서 TD-MPC2 작업자 17개와 같이 돌아 여전히 episode당 75초다. GPU 1로 옮긴 PointMaze는 episode당 10초로 잘 돈다(mujoco-py가 동작함). RunPod: b-dp pod에서 E2 runner 스모크 4/4가 통과했다(첫 시도 오류는 수정됨). b-e0 pod은 반사실 보정과 32B VLM 다운로드 중이다. 누적 $2.20, 감시 스크립트 정상 |
| 15:08 | 서버 GPU 0·1 모두 100%, 디스크 여유 294GB(B 정리로 회복). B의 DP 학습은 epoch 약 700/1000이고 15:40Z 무렵 끝날 예정이다. B의 중간 checkpoint 평가에서 DP 성공률은 epoch 80에 0.30~0.44, epoch 220에 0.60~0.62로, 단계별로 퍼져 있다. **A DINO-WM의 PushT held 평가가 episode당 75초(시간당 약 48개)로 파일럿보다 약 4배 느리다.** GPU 0을 TD-MPC2 작업과 나눠 쓰는 탓인지 확인해 달라고 해당 작업자에게 요청했다. E1 전에 GPU 배정을 다시 정할 예정이다. RunPod: E0b pod 2대(b-dp, b-e0, A40)가 설치를 마쳤고 누적 $0.97이다. 감시 스크립트 정상 |
| 14:45 | B 작업자 보고: B의 작은 진단 작업 3개가 GPU 0(A용)에서 돌았다(13:48~14:17Z, 14:40~14:42Z). 원인은 tmux에 바로 넘긴 명령이 `CUDA_VISIBLE_DEVICES`를 물려받지 못한 것이다. B 스크립트가 GPU 1이 아니면 실행을 거부하도록 고쳤다. 수치에는 영향이 없고, A의 처리량 측정만 그 시간대에 느렸을 수 있어 A 작업자 둘에게 다시 재라고 알렸다. DP checkpoint는 8단계(epoch 20·25·30·40·50·80·220·1000)만 남겨 8.1GB(끝날 때 약 9.5GB)이고, 특징 캐시는 0.33GB로 줄었다. `val nan`은 검증을 건너뛴 epoch의 표시였다(실제 검증값은 모두 유한). 부수 효과로 DINOv2 84MB가 `~/.cache/torch/hub`에 받아졌다(남겨 둠) |
| 14:07 | 서버 GPU 0·1 모두 100%, tmux 5개 정상. 디스크 여유 319 → 273GB(1시간에 46GB, 주로 B의 DP checkpoint 17GB와 특징 13GB). B 작업자에게 EMA 전용 저장과 정리, DP `val nan` 원인 확인을 요청함. A 재현 진행 중(mw-pick-place k3: sh-iter 10/10, sh-ep 4/10). B 판정기 (a) held-out 정확도 0.954(R2-12와 같음). M3 p=0.3 과대 추정 0.032(R2-9의 0.039와 비슷). RunPod b-e0 $0.46, 감시 스크립트 정상 |
