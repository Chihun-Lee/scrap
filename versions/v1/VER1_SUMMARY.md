# ver1 정리 (2026-03 ~ 2026-04)

철스크랩 panoptic segmentation 1차 시도. **논문 baseline(PQ 0.55)에 크게 미달**하며 종료.
이 문서는 ver1에서 무엇을 했고, 왜 실패했는지를 기록해 ver2 설계의 출발점으로 삼는다.

## 1. 무엇을 했나

- **데이터**: 4K(3840×2160) 트럭 화물칸 상부 이미지, Train 2,096 / Val 419 (총 2,515장)
- **라벨**: LabelMe JSON 폴리곤, **원시 89 클래스 → 병합 19 클래스** (`2_remap_labelme_exact.py` 하드코딩 매핑)
- **인스턴스 규모**: train 417,218개 (이미지당 평균 199개, 최대 1,814개)
- **파이프라인**: 전처리 0~4 → YOLO11 seg 학습 → panoptic 변환 6,8 → PQ 평가 9

## 2. 학습 시도와 결과

| 시도 | 설정 | 결과 |
|------|------|------|
| reproduce | yolo11s-seg, imgsz=640, 50ep | Mask mAP50 **0.07** |
| **v2** | yolo11m-seg, imgsz=1280, 100ep, AdamW lr=1e-4, node009 | Mask mAP50 **0.131**, 19개 중 **3개만 사용 가능** |
| v3 (freeze/nano/safe/resume) | 로컬 MPS, yolo11n/s | 실험 중단 |
| RF-DETR seg-nano | 로컬 MPS | 2 epoch 초기 테스트만 |

**v2 상세 (val 419장 기준)**: 전체 Mask mAP50=0.131 / mAP50-95=0.083
- 사용 가능: handler(0.995), square pipe(0.696), mixed steel(0.658)
- 나머지 16개 클래스: 거의 0 (rebar, structure steel, panel, small pipe, trash, vehicle 등 전부 붕괴)

## 3. 실패 원인 (라벨 단계에서 비롯된 것 중심)

1. **극심한 클래스 불균형 (~1600:1)** — structure steel 129,364개 vs plastic 79개.
   소수 클래스가 학습되지 않고 붕괴.
2. **초소형 객체** — YOLO 640 리사이즈 시 8px 미만 객체가 **278,322개** (train 인스턴스의 절반 이상).
   imgsz=1280으로 키워도 이미지당 200개 밀집 객체에는 부족.
3. **89→19 강제 병합의 의미 혼선** — 예: `79. Unknown-Panel` → structure steel,
   `80. Unknown-Square Pipe` → structure steel, `Cabinet`/`Paint Can`도 structure steel로 묶임.
   서로 다른 외형이 한 클래스에 섞여 학습 신호가 모순됨.
   원시 89개 중 일부는 매핑조차 안 됨(`unmapped_mode="keep"`로 통과).
4. **이미지당 ~200개 인스턴스의 라벨 일관성** — 가림/중첩/경계 처리 기준이 불명확.

## 4. ver2로 가져갈 시사점

- 실패의 큰 축은 **모델이 아니라 데이터/라벨**에 있다.
- ver2는 **라벨링 가이드라인 재설계**부터 시작한다:
  - 클래스 체계(병합 기준) 재정의 → 의미 혼선 제거, 불균형 완화
  - 최소 객체 크기 / 군집 객체 처리 규칙
  - thing(셀 수 있는 개체) vs stuff(영역) 구분
  - 가림·경계·누락 허용 기준 명문화
- 외주 업체가 따를 수 있는 **라벨링 가이드라인 문서**가 ver2의 첫 산출물.

## 5. 자산 위치

- `versions/v1/scripts/` — 실제 돌린 전처리(0~4)·학습 스크립트
- `versions/v1/base_weights/` — pretrained base (yolo11n/s-seg, rf-detr-seg-nano)
- `versions/v1/runs/` — 학습 결과물 (가중치·로그, gitignored)
- `data/analysis/output/report.txt` — 데이터 통계 리포트 (클래스 분포·객체 크기)
- `references/pipeline/` — 표준 0~9 파이프라인 (참고용 원본)
- `references/paper/` — 논문 PDF (baseline 출처)
