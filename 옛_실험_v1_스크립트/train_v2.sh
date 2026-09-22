#!/bin/bash
# Steel Scrap Segmentation v2 — 재학습 (node009)
# 개선사항: imgsz 1280, max_det 500, yolo11m-seg, epochs 100

set -e

WORKDIR=/home/chihun/scrap
cd $WORKDIR
export PATH=/home/chihun/miniconda3/envs/scrap/bin:$PATH

echo "============================================"
echo "Steel Scrap Segmentation v2 Training"
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "============================================"

# 학습 실행
# - imgsz 1280: 4K->640 리사이즈에서 객체 소실 방지
# - max_det 500: 이미지당 평균 107개 인스턴스 대응
# - yolo11m-seg: 19 클래스 + 고밀도 장면에 충분한 모델 용량
# - epochs 100: 충분한 수렴
# - batch 4: 1280 해상도에서 98GB VRAM 고려
yolo task=segment mode=train \
    data=datasets/data.yaml \
    model=yolo11m-seg.pt \
    epochs=100 \
    batch=4 \
    imgsz=1280 \
    optimizer=AdamW \
    lr0=0.0001 \
    augment=True \
    max_det=500 \
    device=0 \
    workers=8 \
    project=runs/segment \
    name=v2_m_1280 \
    exist_ok=True

echo "============================================"
echo "Training complete!"
echo "Results: runs/segment/v2_m_1280/"
echo "Best weights: runs/segment/v2_m_1280/weights/best.pt"
echo "============================================"
