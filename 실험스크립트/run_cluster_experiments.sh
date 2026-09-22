#!/bin/bash
# 클러스터 순차 실행 오케스트레이터 — 7/24 itivai 공유 대비 (2026-07-14 작성)
# 사용: 클러스터에서  nohup bash run_cluster_experiments.sh > run_all.log 2>&1 &
# 전제: conda env scrap, ultralytics >= 8.4.90 (yolo26+imgsz1280 이슈 픽스)
set -e
cd "$(dirname "$0")"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate scrap

python -c "import ultralytics as u; v=tuple(map(int,u.__version__.split('.')[:3])); assert v>=(8,4,90), 'ultralytics %s < 8.4.90 — pip install -U ultralytics 필요' % u.__version__"

EPOCHS="${EPOCHS:-100}"
DEVICE="${DEVICE:-0}"

# 1) 2순위: 컷오프 스윕 (무필터/8/10/12/16) — 7/24 공유의 본체
python exp2_train_sweep.py --epochs "$EPOCHS" --device "$DEVICE"

# 2) 세장형 예외 정책 ablation — itivai 7/14 확인요청 3번 학습 검증
#    (cut10은 exp2 가중치를 자동 재사용해 평가만 수행 → 실제 학습은 2조건)
python exp6_exception_ablation.py --epochs "$EPOCHS" --device "$DEVICE"

# 3) 3순위 스크리닝 (7/31 목표) — 시간 남으면 이어서
python exp3_imgsz_points.py --epochs 60 --device "$DEVICE" || true

echo "ALL DONE $(date)"
