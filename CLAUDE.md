# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ⚠ 폴더 구조 재정리 (2026-07-26)

최상위가 목적별 그룹으로 바뀜 — 매핑은 `폴더구조_20260726.md` 참조.
`Project/`·`data/`·`references/`는 실행 경로(launchd·스크립트)가 걸려 있어 **이동 금지**.
아래 본문에 나오는 옛 경로는 이렇게 읽을 것: `datasets.zip`→`데이터_아카이브/`, 메일 자료→`협력_메일/`,
행정 자료→`행정_문서/`, PPT·보고서→`산출물_보고/`.
⚠ **데이터 분할 주의(2026-07-26 발견)**: `data/datasets/val_data` 419장은 `train_data` 2,096장의
바이트 동일 복사본 — 기존 val 평가는 학습셋 위 측정임. 새 실험은 `../GT무필터정확도/`의
그룹 분할(clean) 데이터셋을 쓸 것.

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

## 알림 시스템 (cluster-notify)

학습 완료/오류/결정 필요 시 Telegram 알림 전송:
```python
import sys, os
sys.path.insert(0, os.path.expanduser("~/Code/클러스터/cluster-notify"))
from notify import training_complete, error, decision_needed, wait_for_decision

# 학습 스크립트 끝에 추가
try:
    # ... 학습 코드 ...
    training_complete("scrap", "YOLO11 학습 완료", f"mAP: {map50:.4f}, epochs: {epoch}")
except Exception as e:
    error("scrap", f"학습 오류: {e}")
```

## Target Metrics (Paper Baseline)
- Count Accuracy: 80.2%
- Area Ratio Accuracy: 86.9%
- Panoptic Quality (PQ): 0.55
