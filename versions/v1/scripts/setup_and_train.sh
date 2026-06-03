#!/bin/bash
# Steel Scrap Segmentation 재현 스크립트
# node004에서 실행

set -e

WORKDIR=~/chihunlee/scrap
cd $WORKDIR

echo "============================================"
echo "[1/5] 환경 설정"
echo "============================================"

# Python 가상환경 생성
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

pip install --upgrade pip
pip install ultralytics tqdm pycocotools pandas pillow opencv-python-headless
pip install numpy==1.26.4
pip install "git+https://github.com/cocodataset/panopticapi.git"

echo "============================================"
echo "[2/5] 데이터 압축 해제"
echo "============================================"

if [ ! -d "datasets/train_data" ]; then
    unzip -o datasets.zip -d .
    echo "datasets.zip 압축 해제 완료"
else
    echo "datasets/ 이미 존재 — skip"
fi

# 필요한 디렉토리 생성
mkdir -p datasets/images/train
mkdir -p datasets/images/val
mkdir -p datasets/labels/train
mkdir -p datasets/labels/val
mkdir -p datasets/annotations

echo "============================================"
echo "[3/5] 데이터 전처리 (0~4)"
echo "============================================"

# 스크립트를 yolo 옵션 파일 폴더에서 workdir로 복사
cp -f "yolo 옵션 파일/0_remove_cargo.py" .
cp -f "yolo 옵션 파일/1_remove_small_filter.py" .
cp -f "yolo 옵션 파일/2_remap_labelme_exact.py" .
cp -f "yolo 옵션 파일/3_annotations_to_instances.py" .
cp -f "yolo 옵션 파일/4_labels_yolo.py" .
cp -f "yolo 옵션 파일/5_GT_mask.py" .
cp -f "yolo 옵션 파일/6_instance_to_panoptic.py" .
cp -f "yolo 옵션 파일/7_pred_mask.py" .
cp -f "yolo 옵션 파일/8_yolopred_to_panoptic.py" .
cp -f "yolo 옵션 파일/9_pq_calculator.py" .
cp -f "yolo 옵션 파일/run_data_preprocessing.py" .
cp -f "yolo 옵션 파일/run_pq_eval.py" .

# 전처리 실행
python run_data_preprocessing.py

echo "============================================"
echo "[4/5] YOLO11s-seg 학습"
echo "============================================"

# 논문 조건: epochs=50, batch=4, imgsz=640, optimizer=AdamW, lr=0.0001
# yolo11s-seg.pt pretrained weights 다운로드 (자동)
yolo task=segment mode=train \
    data=datasets/data.yaml \
    model=yolo11s-seg.pt \
    epochs=50 \
    batch=4 \
    imgsz=640 \
    optimizer=AdamW \
    lr0=0.0001 \
    augment=True \
    device=1 \
    workers=4 \
    project=runs/segment \
    name=reproduce

echo "============================================"
echo "[5/5] 학습 완료"
echo "============================================"

echo "학습 결과: runs/segment/reproduce/"
echo "Best weights: runs/segment/reproduce/weights/best.pt"
