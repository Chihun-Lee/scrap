# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

## Target Metrics (Paper Baseline)
- Count Accuracy: 80.2%
- Area Ratio Accuracy: 86.9%
- Panoptic Quality (PQ): 0.55
