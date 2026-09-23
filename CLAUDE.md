# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 폴더 안내 (2026-09-19)

이 저장소는 `철스크랩/2_코드/라벨링기준연구/`에 있다. 전체 지도는 `철스크랩/README.md`.
- `Project/` · `references/`는 실행 경로(launchd · 스크립트)가 걸려 있어 옮기지 않는다
- `data` → `../../1_데이터/원본데이터/20251209_NAS_작년데이터`(심링크). 아래 본문의 `datasets.zip`은 이미 그 폴더에 풀려 있다
- 메일은 `철스크랩/4_문서/3_메일_itivai/`, 행정은 `_삭제예정_20260923_2차/3_행정_이전담당자/행정/`, PPT · 보고서는 `4_문서/2_발표자료/`
- ⚠ `data/datasets/val_data` 419장은 `train_data` 2,096장을 복사한 것이다. 새 실험은 `../2_GT무필터정확도/`의 `clean` 세트를 쓴다. 점수는 박스 기준(box mAP50)으로 적는다

## Project Overview

Steel scrap panoptic segmentation project — classifies steel scrap types and individual objects in truck cargo images using YOLO11 instance segmentation, then evaluates with Panoptic Quality (PQ) metrics. Based on the paper "Steel Scrap Segmentation via Panoptic Segmentation Approach" (Transactions of Materials Processing, Vol.34 No.6, 2025).

## Data

- **Source**: 4K (3840×2160) images of truck cargo tops from steel scrap yards
- **Split**: Train 2,096 / Val 419 images (total 2,515)
- **Format**: LabelMe JSON polygons (89 raw classes → 19 merged classes)
- **Location**: `datasets.zip` (8.2GB) extracts to `datasets/train_data/` and `datasets/val_data/`
- Cropped versions exist (`train_cropped.zip`, `val_cropped.zip`) but primary work uses original data

## Pipeline

All scripts live in `yolo 옵션 파일/` and must be copied to the working directory before execution. They use relative paths assuming execution from the project root.

### Preprocessing (run via `python run_data_preprocessing.py`)
```
0_remove_cargo.py      — Remove "Cargo Area" labels from JSONs (paths: datasets/train_data/, datasets/val_data/)
1_remove_small_filter.py — Filter objects < 8px after YOLO 640 resize, copy images to datasets/images/{split}
2_remap_labelme_exact.py — Remap 89 raw labels → 19 merged classes, output to datasets/{split}_remapped/
3_annotations_to_instances.py — LabelMe → COCO instance JSON (datasets/annotations/instances_{split}.json)
4_labels_yolo.py       — COCO → YOLO seg format (datasets/labels/{split}/), generates data.yaml and classes.txt
```

### Training
```bash
yolo task=segment mode=train data=datasets/data.yaml model=yolo11s-seg.pt epochs=50 batch=4 imgsz=640 optimizer=AdamW lr0=0.0001 device=0
```

### Evaluation (run via `python run_pq_eval.py`)
```
5_GT_mask.py           — Visualize GT annotations with color overlay
6_instance_to_panoptic.py — Convert GT instances → panoptic format (PQ/panoptic_test/)
7_pred_mask.py         — Generate prediction color overlay from YOLO weights
8_yolopred_to_panoptic.py — Convert YOLO prediction txt → panoptic format (PQ/panoptic_pred/)
9_pq_calculator.py     — Compute PQ using panopticapi (GT vs Pred)
```

### Key paths after preprocessing
```
datasets/images/{train,val}/     — images
datasets/labels/{train,val}/     — YOLO format labels
datasets/annotations/            — COCO instance JSONs
datasets/data.yaml               — YOLO dataset config
PQ/                              — panoptic evaluation outputs
runs/segment/                    — training outputs
```

## 19 Merged Classes
handler, rebar, structure steel, mixed steel, heavy iron, panel, square pipe, mesh, small pipe, trash, vehicle, pipe, plastic, machine, LPG GAS cylinder, beam, drum, Fan, Guillotine

## Remote Execution

Code is written on MacBook and synced to GPU cluster via `sync-to-cluster.sh`. Training runs on cluster nodes (node002 has Python 3.10 + RTX A6000 × 8). The cluster uses a shared home directory (`/home/user/chihunlee/scrap/`).

## 실행 환경

- **회사 와이파이** (클러스터 접속 가능): 클러스터에서 학습
- **집 와이파이** (클러스터 접속 불가): 맥북 로컬 실행 (`device = "mps"`, M4 Pro 48GB)
- **Anaconda 사용 금지** — miniforge(conda-forge)만 사용

## 알림

학습 스크립트는 알림을 직접 보내지 않는다. 끝나면 결과 CSV와 `.done` 파일만 남기고, 확인은 그 파일로 한다(`from notify import …`는 쓰지 않는다).

## Target Metrics (Paper Baseline)
- Count Accuracy: 80.2%
- Area Ratio Accuracy: 86.9%
- Panoptic Quality (PQ): 0.55
