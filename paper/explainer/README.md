# B 연구설명 문서 (Near-Miss Blindness 연구)

DriveWAM 연구설명(`research/papers/DriveWAM_Lightweight_Action_Readout/…_연구설명.pdf`)과 같은 형식으로 만든 한국어 설명 문서다. 2026-10-26 작성.

- 구성: 요약(핵심 결과 표, 핵심 발견) → 1. 배경지식 → 2. 측정 방법 → 3. 실험 설정 → 4. 결과 → 5. 해석과 한계. 부록과 다음 단계는 사용자 요청으로 뺐다.
- 주장 범위: `main/DESIGN_main.md` §8·§10·§11, `main/reports/FINAL_B.md` §3, JKAIA 포스터 최종본(10-25)을 따른다.
- 글꼴: Apple SD Gothic Neo(산돌고딕).

## 파일

| 경로 | 내용 |
|---|---|
| `build/content.py` | 원고. 블록 목록 하나로 Word와 PDF를 함께 만든다 |
| `build/make_figs.py` | 그림 12개. 개념도는 직접 그리고, 결과 그림은 `main/results`의 분석 JSON만 읽는다(새 계산 없음) |
| `build/build.py` | Word(.docx)는 python-docx로, PDF는 같은 원고를 HTML로 바꿔 Chrome으로 인쇄한다 |
| `build/verify_numbers.py` | 원고 수치 225개를 결과 JSON에서 다시 계산해 대조한다 |
| `out/` | `Near-Miss_Blindness_연구설명.{docx,pdf,html}` |

- 최종 docx·pdf는 `research/papers/Near-Miss Blindness/`에 복사했다.

## 다시 만들기

```bash
cd "/Users/song-yonghwi/Documents/physical ai - Neural Networks/B_near_miss_blindness/paper/explainer/build" && python3 make_figs.py && python3 verify_numbers.py && python3 build.py
```

- PDF를 Chrome으로 만드는 이유: Word를 AppleScript로 PDF 저장하면 Microsoft 변환 경로를 거쳐 Apple SD Gothic Neo가 맑은 고딕·Cambria로 바뀐다(글꼴을 문서에 포함해도 같았다). Word에서 직접 'PDF로 저장 → 인쇄에 최적화'를 고르면 산돌고딕이 유지된다.
- Apple SD Gothic Neo에는 U+2212(−), 아래첨자 숫자 ₀·₅, ŝ가 없다. 원고는 음수에 en dash(–)를 쓰고 r_{50} 같은 표기는 아래첨자 서식으로 쓴다. ŝ는 대체 글꼴로 그려진다.
- `make_figs.py`는 시스템 TTC에서 글꼴 면을 꺼내 임시 폴더(`B_EXPLAINER_FONT_CACHE` 또는 시스템 임시 폴더)에 둔다.
