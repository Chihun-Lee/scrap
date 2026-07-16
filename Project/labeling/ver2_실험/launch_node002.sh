#!/bin/bash
# node002 병렬 실행 — exp2 컷 5조건(GPU0-4) + exp6 예외 2조건(GPU5-6)  (2026-07-15)
# 사용: node002에서  bash launch_node002.sh
# 알림: Telegram 폐기 — 완료는 exp{2,6}_results.csv 행 수를 모니터링 세션이 폴링해 Claude 앱 푸시.
set -e
cd "$(dirname "$0")"
source ~/miniforge3/etc/profile.d/conda.sh
conda activate scrap

EPOCHS="${EPOCHS:-100}"
BATCH="${BATCH:-4}"   # A6000 48GB, yolo26x-seg@1280 추정 17~25GB — 무인 실행이라 고정 배치

# 1) 데이터셋 변형 선생성 (병렬 프로세스 간 생성 경합 방지, 이미 있으면 스킵)
[ -f datasets_exp/base/data.yaml ]        || python prepare_yolo_dataset.py --out base        --min-sqrt-area 0
[ -f datasets_exp/cut8/data.yaml ]        || python prepare_yolo_dataset.py --out cut8        --min-sqrt-area 8
[ -f datasets_exp/cut10/data.yaml ]       || python prepare_yolo_dataset.py --out cut10       --min-sqrt-area 10
[ -f datasets_exp/cut12/data.yaml ]       || python prepare_yolo_dataset.py --out cut12       --min-sqrt-area 12
[ -f datasets_exp/cut16/data.yaml ]       || python prepare_yolo_dataset.py --out cut16       --min-sqrt-area 16
[ -f datasets_exp/cut10_noexc/data.yaml ] || python prepare_yolo_dataset.py --out cut10_noexc --min-sqrt-area 10 --keep-elongated 0
[ -f datasets_exp/cut10_w2/data.yaml ]    || python prepare_yolo_dataset.py --out cut10_w2    --min-sqrt-area 10 --min-elongated-width 2

# 2) 학습 7개 병렬 기동 (GPU 1장씩)
nohup python exp2_train_sweep.py --cuts 0  --device 0 --epochs "$EPOCHS" --batch "$BATCH" > exp2_base.log  2>&1 &
nohup python exp2_train_sweep.py --cuts 8  --device 1 --epochs "$EPOCHS" --batch "$BATCH" > exp2_cut8.log  2>&1 &
nohup python exp2_train_sweep.py --cuts 10 --device 2 --epochs "$EPOCHS" --batch "$BATCH" > exp2_cut10.log 2>&1 &
nohup python exp2_train_sweep.py --cuts 12 --device 3 --epochs "$EPOCHS" --batch "$BATCH" > exp2_cut12.log 2>&1 &
nohup python exp2_train_sweep.py --cuts 16 --device 4 --epochs "$EPOCHS" --batch "$BATCH" > exp2_cut16.log 2>&1 &
nohup python exp6_exception_ablation.py --variants cut10_noexc --device 5 --epochs "$EPOCHS" --batch "$BATCH" > exp6_noexc.log 2>&1 &
nohup python exp6_exception_ablation.py --variants cut10_w2    --device 6 --epochs "$EPOCHS" --batch "$BATCH" > exp6_w2.log    2>&1 &

sleep 5
echo "=== 기동된 프로세스 ==="
pgrep -af "exp[26]_" | head -10
echo "런처 완료 $(date)"
