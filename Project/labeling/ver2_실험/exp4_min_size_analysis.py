# -*- coding: utf-8 -*-
"""4순위: 클래스별 크기 구간(size-bin) 검출 성능 분석 → 클래스별 최소 유의미 크기 도출.

학습된 가중치로 val 전체를 추론하고, GT 인스턴스를 sqrt(area)@1280 크기 구간별로 나눠
클래스×구간 recall을 계산한다. "recall이 대형 구간 대비 급락하는 지점"이 해당 클래스의
최소 유의미 크기 후보가 된다.

사용 (클러스터, 2·3순위 학습 완료 후):
  python exp4_min_size_analysis.py --weights runs/exp2_cut8_.../weights/best.pt
  python exp4_min_size_analysis.py --weights best.pt --variant base   # 무필터 GT 기준(소형 포함 전체)
"""
import argparse
import csv
import glob
import math
import os
from collections import defaultdict

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BINS = [(0, 4), (4, 6), (6, 8), (8, 10), (10, 12), (12, 16), (16, 24), (24, 32), (32, 1e9)]
IOU_THR = 0.5
GRID_SCALE = 1 / 3.0  # 3840×2160 → 1280×720 그리드에서 마스크 IoU 계산


def bin_label(v):
    for lo, hi in BINS:
        if lo <= v < hi:
            return "{:g}-{:g}".format(lo, hi) if hi < 1e9 else ">={:g}".format(lo)
    return "?"


def poly_mask(poly, w, h):
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(m, [np.round(np.array(poly) * GRID_SCALE).astype(np.int32)], 1)
    return m


def mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    return inter / (np.logical_or(a, b).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--variant", default="base", help="GT 기준 데이터셋 변형 (기본 base=무필터)")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    from ultralytics import YOLO
    ds = os.path.join(HERE, "datasets_exp", args.variant)
    classes = []
    in_names = False
    for line in open(os.path.join(ds, "data.yaml"), encoding="utf-8"):
        if line.startswith("names:"):
            in_names = True
        elif in_names and line.strip().startswith("- "):
            classes.append(line.strip()[2:])

    model = YOLO(args.weights)
    n_gt = defaultdict(int)      # (class, bin) → GT 수
    n_hit = defaultdict(int)     # (class, bin) → 매칭 수

    val_imgs = sorted(glob.glob(os.path.join(ds, "images", "val", "*.jpg")))
    for img_path in val_imgs:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(ds, "labels", "val", stem + ".txt")
        if not os.path.exists(lbl_path):
            continue
        img = cv2.imread(img_path)
        H, W = img.shape[:2]
        gw, gh = round(W * GRID_SCALE), round(H * GRID_SCALE)

        # GT 로드 (정규화 폴리곤 → 원본 좌표)
        gts = []  # (cls, poly, sqrt_area@1280)
        for line in open(lbl_path):
            parts = line.split()
            if len(parts) < 7:
                continue
            c = int(parts[0])
            xy = np.array(parts[1:], dtype=np.float64).reshape(-1, 2) * [W, H]
            area = abs(cv2.contourArea(xy.astype(np.float32)))
            sa = math.sqrt(area) * (args.imgsz / max(W, H))
            gts.append((c, xy, sa))

        # 예측
        r = model.predict(img, imgsz=args.imgsz, conf=args.conf, max_det=1000,
                          device=args.device, verbose=False)[0]
        preds = defaultdict(list)  # cls → [(conf, mask)]
        if r.masks is not None:
            for k in range(len(r.boxes)):
                c = int(r.boxes.cls[k])
                preds[c].append((float(r.boxes.conf[k]), poly_mask(r.masks.xy[k], gw, gh)))

        # 클래스별 greedy 매칭 (conf 내림차순)
        for c in set(g[0] for g in gts):
            g_list = [(i, poly_mask(g[1], gw, gh), g[2]) for i, g in enumerate(gts) if g[0] == c]
            used = set()
            for conf, pm in sorted(preds.get(c, []), key=lambda t: -t[0]):
                best_iou, best_i = 0.0, -1
                for i, gm, _ in g_list:
                    if i in used:
                        continue
                    iou = mask_iou(pm, gm)
                    if iou > best_iou:
                        best_iou, best_i = iou, i
                if best_iou >= IOU_THR:
                    used.add(best_i)
            for i, _, sa in g_list:
                b = bin_label(sa)
                n_gt[(c, b)] += 1
                if i in used:
                    n_hit[(c, b)] += 1

    out = os.path.join(HERE, "exp4_recall_by_size.csv")
    bins = [bin_label((lo + min(hi, 100)) / 2) for lo, hi in BINS]
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["class"] + [b + " n" for b in bins] + [b + " recall" for b in bins])
        for c, name in enumerate(classes):
            ns = [n_gt.get((c, b), 0) for b in bins]
            rs = [round(n_hit.get((c, b), 0) / n, 3) if n else "" for b, n in zip(bins, ns)]
            w.writerow([name] + ns + rs)
    print("saved ->", out)


if __name__ == "__main__":
    main()
