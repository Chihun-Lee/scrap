# 1순위 검토 — 입력 1280 기준 Polygon Point 수 기준

> **검토 주체**: 재료연구원 / **담당(도출)**: ITV AI, 목표 6/12 (메일② 기준)
> **목적**: ITV가 도출할 Point 수 기준을 우리가 사전 검토하고, 공학적 근거·권고값·검증방법을 제시.
> **근거**: `공부자료/02_Polygon_Point수.md`, `공부자료/분석_라벨링방안_타당성_및_1순위검토.md` + 요청서(수정본 0528 §3) + 현재 파이프라인 코드.
> 표기 **[F]** 검증된 사실 / **[I]** 사실로부터의 추론(실측 필요). 작성 2026-06-03.

---

## 1. 요청 배경 (요청서 §3 + 메일②)

- **문제 제기(요청서)**: 현재 라벨은 4K 원본 기준 Polygon Point가 매우 세밀 → 이미지당 Instance·Point 수 과다 → 학습 시 **GPU 메모리 과도 증가**.
- **요청**: 입력 이미지 크기 **1280** 기준에서 **학습 가능한 Point 수 기준**(= 업체가 polygon 작성/보정 시 참고할 **Point 수 상한**)을 도출.
- **일정/담당(메일②)**: 라벨링 착수(6/17) 전 선확보 필요 → **ITV AI가 직접 6/12까지** 러프 기준 도출. 우리는 검토·교차확인.
- **확인 요청 사항 6개**: ① 1280서 YOLO26x-seg 학습 가능 여부 ② OOM 안 나는 Point 수 ③ Instance당 최대 Point ④ 이미지 1장당 총 Point ⑤ Point 단순화 수준별 Mask 품질 ⑥ Point 단순화 수준별 학습 가능 여부·GPU 메모리.

---

## 2. 현재 코드 실태 (확인됨)

- `references/pipeline/4_labels_yolo.py` (= `Project/segmentation/versions/v1/scripts/4_labels_yolo.py`) **99~110행**: COCO polygon의 **모든 점을 단순화 없이** 정규화해 YOLO 라벨로 기록.
- 즉 **4K 세밀 polygon의 vertex가 그대로 학습 파이프라인에 전달** → 요청서가 지적한 "과도한 Point"가 그대로 통과. **단순화(상한) 적용 지점이 코드에 부재.**
- 같은 파일 81행: `iscrowd=1`은 skip → 향후 dense pile용 ignore 영역 도입 시 별도 처리 필요(1순위 범위 밖, §6 참고).

---

## 3. 결정적 기술 근거 — YOLO-seg가 "학습할 수 있는" 마스크 디테일의 상한

**[F]** YOLO11/26-seg 마스크는 **prototype(stride 4)** 기반:
- prototype grid = **입력 / 4**. → imgsz 640 → 160×160 [F], **imgsz 1280 → 320×320** [I, /4 규칙 직접 적용].
- loss는 `mask_ratio=4`로 한 번 더 다운샘플 → **약 80×80 grid에서 supervision** [F].

**[F]** 해상도 매핑: 3840×2160 → 1280 의 스케일 **s = 1280/3840 = 1/3**
- **입력 1px = 원본 3px**
- **proto cell 1개(입력 4px) ≈ 원본 12px**

**[I] 핵심 함의**:
> polygon vertex 간격이 **proto cell(원본 ~12px / 입력 ~4px)보다 촘촘하면 학습 신호가 0**이다. supervision grid(~80×80)는 더 거칠어 더더욱 그렇다.
> → 그보다 세밀한 vertex는 **mask 품질을 못 올리고**, 라벨 용량·dataloader rasterization·augmentation 비용만 증가시킨다.

이것이 "Point 수 상한"이 공학적으로 정당한 근본 이유다.

---

## 4. 권고 Point 수 기준 (학회 논문 근거)

### 4.1 per-instance 기준
| 구분 | 권고값 | 근거 |
|------|--------|------|
| **Hard ceiling** | **128 vertex** | Deep Snake [F]: 128이면 충분, **192는 오히려 성능 저하** |
| **전형적 필요 수** | 대형 **30–40**, 소형 **<10** | Polygon-RNN++/Curve-GCN 30–40 [F], PolarMask 36 ray [F] |
| **원칙 규칙(자동)** | **vertex ≤ ceil(둘레@1280 / 4)** | proto cell = 입력 4px [I] |

### 4.2 단순화 수단 (per-image 총 Point는 이걸로 자연 수렴)
- **Douglas–Peucker** (`cv2.approxPolyDP`) [F], **ε ≈ 1 proto cell**:
  - 입력 1280 좌표계: **ε ≈ 3–4 px**
  - 원본 4K 좌표계: **ε ≈ 10–12 px**
- + per-instance **128 hard cap** 병행.
- **[I]** 이 단순화에서 supervised IoU 손실 ~0 (디테일이 어차피 학습 grid 이하이므로).

### 4.3 per-image 총 Point (거친 order 추정) [I]
- ver1 평균 ~199 instance/image. instance당 평균 ~20–40 vertex로 단순화 시 → **이미지당 ~4,000–8,000 point** 수준.
- 현재(미단순화) 4K 세밀 라벨은 이보다 수배~수십배 → 여기서 I/O·전처리·OOM 비용 발생.
- ※ 절대 상한을 못 박기보다 **per-instance 규칙(§4.1+§4.2)을 걸면 총량은 자동 수렴**하는 것이 견고.

---

## 5. 확인 요청 사항 6개 — 직답

| # | 항목 | 답변 |
|---|------|------|
| ① | 1280서 YOLO26x-seg 학습 가능? | **[I]** instance 수·batch에 의존. Point cap + 소형 instance 컷오프(2·4순위) 병행 시 가능성 높음. 실측 필요. |
| ② | OOM 안 나는 Point 수 | **[I]** per-instance RDP(ε≈1 proto cell)+128 cap으로 bound. 절대 임계는 instance 수와 결합해 실측. |
| ③ | Instance당 최대 Point | **128 hard [F] / 실무 40–60 권장 [I]** |
| ④ | 이미지 1장당 총 Point | **[I]** §4.3 — RDP로 자연 수렴(목표 수천 단위). 별도 절대상한 비권장. |
| ⑤ | Point 단순화별 Mask 품질 | **[I]** ε ≤ 1 proto cell까지 IoU 손실 ~0, 초과 시 급락 예상 → **ε–IoU 곡선 실험으로 확정**(§7). |
| ⑥ | Point 단순화별 학습가능·GPU | **[I]** vertex↓ → 라벨 I/O·전처리·augment 비용↓(주 효과). **GPU activation은 vertex보다 instance 수·마스크 해상도가 지배**(§6). |

---

## 6. ⚠️ 공학적 쟁점 — "Point 수"가 GPU OOM의 *주* 동인이 아닐 수 있음

요청서는 "Point 수 많음 → GPU 메모리 과도"로 기술하나, YOLO-seg **학습 GPU 메모리의 실제 주동인**은 [F/I]:
1. **이미지당 instance 수** (target mask 채널·매칭 비용) ← 가장 큼
2. **마스크 grid 해상도** (∝ imgsz²)
3. batch · 모델 크기 · imgsz

Polygon **vertex 수**가 주로 영향을 주는 곳은 ① 라벨 파일 크기, ② dataloader CPU rasterization, ③ augmentation 변환 — **GPU activation 메모리엔 부차적**이다.

**→ 시사점 (ITV 회신 포인트)**:
- Point 상한은 여전히 타당하다(품질 무손실 단순화 + 저장·전처리 절감 + 안전마진).
- 다만 **진짜 OOM 완화는 instance 수 폭증을 잡는 2·4순위(소형 instance 컷오프)와 묶어야** 효과가 크다.
- **1순위(Point)와 2·4순위(크기)는 독립 변수가 아니라 연동** — Point 기준만으로 OOM이 해결된다고 보면 안 됨.

---

## 7. 검증 실험 설계 제안 (확인사항 ⑤⑥ 확정용)

우리(또는 ITV)가 러프 기준을 수치로 확정하기 위한 최소 실험:

1. **ε–IoU 곡선** (확인사항 ⑤):
   - ver1 GT polygon에 RDP를 ε ∈ {0(원본), 1, 2, 4, 8, 16 px@1280} 적용.
   - 각 ε에서 **단순화 polygon vs 원본 polygon의 mask IoU** + **vertex 수 감소율** 측정.
   - 예상: ε ≈ 1 proto cell(~3–4px@1280)까지 IoU≈1.0 유지, 이후 하락 → 무손실 상한 확정.
2. **Point sweep 학습** (확인사항 ①②⑥):
   - 동일 데이터를 ε∈{0, 1cell, 2cell}로 단순화한 3벌로 YOLO-seg(또는 26x) 학습.
   - **최대 batch / peak GPU 메모리 / 학습 가능 여부 / Mask mAP** 비교.
3. 가능하면 **instance 수 컷오프를 교차변수**로 추가(§6 검증).

---

## 8. 적용 방법 (코드)

- **삽입 지점**: `4_labels_yolo.py` 99~110행의 polygon 기록 직전, 또는 `3_annotations_to_instances.py`(LabelMe→COCO)의 polygon 작성 단계.
- **로직**: 각 polygon에 `cv2.approxPolyDP(poly, epsilon=ε, closed=True)` 적용(ε = 원본 좌표 ~10–12px) → 점수 128 초과 시 ε 증가 재적용(또는 균등 down-sample).
- **라벨링 업체 가이드**: 업체엔 "원본 4K 기준 인접 vertex 간격 ~10px 이상, instance당 최대 ~128점"을 **상한 가이드**로 전달(하한 강제 아님 — 곡률 큰 경계엔 더 촘촘 허용).

---

## 9. 결론

1순위 "Point 수 상한" 방향은 **공학적으로 타당**하며, 근거 있는 수치까지 제시 가능:
- **per-instance ≤ 128 (hard) / 40–60 (실무)**, **RDP ε ≈ 원본 10–12px(=입력 3–4px=1 proto cell)**.
- 근본 이유: **YOLO-seg가 학습하는 마스크 디테일이 prototype grid(1280서 320×320, loss ~80×80)로 제한**되어, 그 이상 세밀한 vertex는 학습 신호 없이 비용만 발생.

**ITV 회신 시 함께 전달할 2가지**:
1. Point 기준은 **무손실 단순화(ε≈1 proto cell)** 로 잡되, ε–IoU 곡선으로 확정 권고.
2. **OOM의 더 큰 동인은 instance 수** — 1순위는 2·4순위(소형 컷오프)와 연동해 봐야 실효.
