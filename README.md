# Steel Scrap Segmentation

철스크랩 panoptic segmentation 프로젝트 — YOLO11 기반 인스턴스 세그멘테이션 + Panoptic Quality 평가

## 개요

트럭 화물칸 상부 촬영 이미지에서 철스크랩 종류별 개별 객체를 분류하는 프로젝트입니다. 논문 "Steel Scrap Segmentation via Panoptic Segmentation Approach" (Transactions of Materials Processing, Vol.34 No.6, 2025)를 기반으로 합니다.

## 데이터

- 4K (3840×2160) 트럭 화물칸 상부 이미지
- Train 2,096 / Val 419 (총 2,515장)
- LabelMe JSON 폴리곤 (89 원시 클래스 → 19 병합 클래스)

## 19 분류 클래스

handler, rebar, structure steel, mixed steel, heavy iron, panel, square pipe, mesh, small pipe, trash, vehicle, pipe, plastic, machine, LPG GAS cylinder, beam, drum, Fan, Guillotine

## 파이프라인

### 1. 전처리
```bash
python run_data_preprocessing.py
```
- `0_remove_cargo.py` — "Cargo Area" 라벨 제거
- `1_remove_small_filter.py` — 640px 리사이즈 시 8px 미만 객체 필터링
- `2_remap_labelme_exact.py` — 89 → 19 클래스 매핑
- `3_annotations_to_instances.py` — LabelMe → COCO 인스턴스 JSON
- `4_labels_yolo.py` — COCO → YOLO seg 포맷

### 2. 학습
```bash
conda activate scrap
yolo task=segment mode=train data=datasets/data.yaml \
     model=yolo11s-seg.pt epochs=50 batch=4 imgsz=640 \
     optimizer=AdamW lr0=0.0001 device=0
```

### 3. 평가 (Panoptic Quality)
```bash
python run_pq_eval.py
```

## 목표 지표 (논문 베이스라인)

| 지표 | 값 |
|------|-----|
| Count Accuracy | 80.2% |
| Area Ratio Accuracy | 86.9% |
| Panoptic Quality (PQ) | 0.55 |

## 환경

- Python 3.11, PyTorch (CUDA)
- ultralytics (YOLO11), pycocotools, panopticapi
