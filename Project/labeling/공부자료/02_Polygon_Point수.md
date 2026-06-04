# 02. Polygon Vertex(Point) 수 & 마스크 표현 — 1순위 근거자료

> 목적: imgsz=1280 YOLO-seg 학습에서 방어 가능한 polygon point 수 상한 도출.
> [VERIFIED]=출처 확인, [INFERENCE]=검증된 사실로부터의 추론. 작성: 2026-06-03.

---

## ★ 핵심 기술 노트: YOLO-seg 마스크 해상도 (상한을 결정하는 사실)

YOLOv8/YOLO11-seg 마스크 생성 방식 (Ultralytics source/docs/issue 검증):
- seg head의 `Proto` 모듈은 **P3(stride 8)**에 붙고 `ConvTranspose2d`로 ×2 업샘플 → **net mask stride = 4** (prototype 마스크 = 입력의 **1/4 해상도**). [VERIFIED] (Proto code; YOLOv5 PR #10108)
- 기본 protos `npr=256`, mask 계수 `nm=32`. 최종 마스크 = 32 계수 × prototype → sigmoid → box crop. [VERIFIED]
- **imgsz 640 → prototype 160×160** (640/4). [VERIFIED]
- proto 크기는 입력에 비례 (입력/4) — 416×640류 입력서 `1×32×64×104` 확인. [VERIFIED] (ultralytics #2953)
- 따라서 **imgsz 1280 → prototype grid 320×320** (1280/4). [INFERENCE — /4 규칙 직접 적용]
- `mask_ratio`(기본 **4**)는 **loss 계산 시 target mask에 적용되는 추가 다운샘플**. → imgsz 1280 loss는 **대략 80×80 grid**(320/4)에서 계산. (mask_ratio가 사실상 4로 고정되는 버그 보고도 있음) [VERIFIED] (#1458, #4369, #20200)

**상한 결정 함의 [INFERENCE]:**
네트워크가 *학습·출력*할 수 있는 마스크 디테일은 proto grid(1280서 320×320)가 한계이고, *supervision*은 더 거침(~80×80). 4K(3840폭)→1280은 ~3배 다운스케일 → **proto cell 1개 ≈ 입력 4px ≈ 원본 ~12px**. **proto cell보다 촘촘한 polygon vertex는 학습 신호가 없고 rasterization/메모리 비용(=OOM 원인)만 증가.** 원칙적 상한 = `vertex ≤ ceil(둘레@1280 / 4)`. 실무: Douglas–Peucker tolerance ≈ 1 proto cell(~3–4px@1280 ≈ ~10px@4K)로 단순화 시 supervised IoU 손실 사실상 0.

---

## 핵심 논문 — vertex 수 / contour 표현 증거

### 1. Polygon-RNN — Castrejón et al. — CVPR 2017 — https://arxiv.org/abs/1704.05548 [VERIFIED]
instance annotation을 순차적 vertex 예측으로 재정의. 희소 vertex로 human급 IoU(Cityscapes 78.4%, 거의 inter-annotator 수준) 달성 → 마스크에 dense contour 불필요.

### 2. Polygon-RNN++ — Acuna et al. — CVPR 2018 — https://arxiv.org/abs/1803.09693 [VERIFIED]
실제 객체 polygon은 **"30–40 point" 범위**라고 명시 — "전형적 instance가 필요로 하는 vertex 수"의 직접 인용 가능한 경험적 상한. ~40 초과는 모델 해상도와 싸우는 영역.

### 3. Curve-GCN — Ling et al. — CVPR 2019 — https://arxiv.org/abs/1903.06874 [VERIFIED]
GCN으로 모든 control point 동시 예측. **~40 control point**로 DeepLab pixel 마스크 *능가*. ~40 well-placed vertex가 dense pixel 마스크와 동등/우월.

### 4. Deep Snake — Peng et al. — CVPR 2020 (oral) — https://arxiv.org/abs/2001.01629 [VERIFIED]
**가장 직접적 근거**: "**128 vertices are enough to represent the contour... 더 많이(192) 샘플하면 오히려 성능 저하.**" → **~128 초과는 이득 없고 해로울 수 있는 hard ceiling.** 대형 instance 상한이 128, 소·중형은 훨씬 적음.

### 5. PolarMask — Xie et al. — CVPR 2020 — https://arxiv.org/abs/1909.13226 [VERIFIED]
각 마스크를 **36 ray(36 vertex)**로 표현. COCO mask AP 32.9 — 전체 데이터셋 instance seg에서도 매우 낮은 고정 vertex 수가 유효함을 증명.

### 6. PointRend — Kirillov et al. — CVPR 2020 — https://arxiv.org/abs/1912.08193 [VERIFIED]
경계 정보는 *희소·적응적* point에 집중, 내부 평탄 영역은 디테일 거의 불필요 → vertex 예산을 고곡률 경계에 쓰라(=Douglas–Peucker가 남기는 것).

### 7. BoundaryFormer — Lazarow et al. — CVPR 2022 — [VERIFIED](openaccess CVPR2022)
고정 수 polygon point를 미분가능 rasterize, **mask space에서 supervision** → polygon 디테일은 rasterization 해상도가 상한. 우리 YOLO 상황과 동형: grid보다 촘촘한 vertex는 낭비 = vertex cap의 핵심 논거.

### 8. PolyTransform — Liang et al. — CVPR 2020 — https://arxiv.org/abs/1912.02801 [VERIFIED]
고정 cardinality vertex set deform으로 top 경계 정확도 → capped vertex budget은 표준이지 타협 아님.

### 9. PolyFormer — Liu et al. — CVPR 2023 — [VERIFIED, 단 정확한 N은 미확인]
마스크를 짧은 polygon 시퀀스로 잘 표현됨을 현대적으로 재확인.

---

## 단순화 알고리즘 & 어노테이션 포맷

### 10. Ramer–Douglas–Peucker 단순화 — Douglas&Peucker, Cartographica 1973 / OpenCV `cv2.approxPolyDP` [VERIFIED]
chord에서 perpendicular 거리 > ε인 점만 재귀 보존. ε 하나로 vertex 수↔충실도 trade.
→ **우리 상한 강제 도구**. **ε ≈ 1 proto cell (~3–4px@1280 ≈ ~10–12px@4K)**. YOLO label 생성 전 polygon별 적용. + instance당 절대 vertex ~128 cap(Deep Snake). 실무 관행은 RDP로 **≤50 point** cap도 흔함.

### 11. COCO polygon 저장 포맷 [VERIFIED]
`iscrowd=0`은 `segmentation`을 flat polygon 리스트로 저장(vertex 수 무제한), `iscrowd=1`은 RLE. **포맷이 cap을 강제하지 않음 → 변환 시 우리가 강제해야**(RDP). loss가 rasterized grid에서 돌므로 ~1 proto cell 이상 vertex 밀도는 순수 overhead.

---

## 결론 (1순위 의사결정)
1. **per-instance hard ceiling: ~128 vertex** — Deep Snake [VERIFIED]: 128 saturate, 192 *저하*. 대부분 scrap은 훨씬 적음(PolarMask 36, Polygon-RNN++/Curve-GCN 30–40 [VERIFIED]).
2. **원칙적 규칙: vertex ≤ ceil(둘레@1280 / 4)** (proto grid 320×320 @1280) [INFERENCE]. 더 촘촘한 디테일은 학습 불가·supervision 없음(loss grid는 ~80×80로 더 거침 [VERIFIED]).
3. **수단: Douglas–Peucker(`cv2.approxPolyDP`)** ε≈1 proto cell + 128 hard cap. per-image 총 point 수(=OOM 원인) 직접 공략, supervised IoU는 유지.

적용 위치: `3_annotations_to_instances.py`(LabelMe→COCO), `4_labels_yolo.py`(COCO→YOLO seg)의 polygon 기록 단계에 RDP 삽입.
