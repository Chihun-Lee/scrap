import os
from panopticapi.evaluation import pq_compute
from panopticapi.utils import rgb2id
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

# 경로 설정
gt_json = 'PQ/panoptic_test.json'
gt_folder = 'PQ/panoptic_test/'

pred_json = 'PQ/panoptic_predictions.json'
pred_folder = 'PQ/panoptic_pred/'

# PQ 계산
print("📊 PQ 계산 중...")

pq_res = pq_compute(
    gt_json_file=gt_json,
    gt_folder=gt_folder,
    pred_json_file=pred_json,
    pred_folder=pred_folder,
)

print("\n✅ PQ 계산 완료.")
print(f"PQ 전체 평균: {pq_res['All']['pq']:.3f}")
print(f"PQ (Thing): {pq_res['Things']['pq']:.3f}")
print(f"PQ (Stuff): {pq_res['Stuff']['pq']:.3f}")
