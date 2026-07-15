# -*- coding: utf-8 -*-
"""stride-4 GT 마스크 생존 시뮬레이션 — itivai 확인요청(2026-07-14) 대응.

YOLO26x-seg @1280 학습 시 seg loss는 GT 폴리곤을 fillPoly@1280 → cv2.resize(1/4,
INTER_LINEAR)로 만든 320그리드 마스크(mask_ratio=4, ultralytics data/utils.py
polygon2mask)와 비교한다. 이 스크립트는 그 파이프라인을 그대로 재현해, 인스턴스별로
"stride 4 feature map에서 형상이 유지되는가"를 전수 측정한다. GPU 불필요(로컬 실행 가능).

  python exp5_stride4_survival.py                 # 전수 (train+val, 약 50만 개)
  python exp5_stride4_survival.py --limit 200     # 스모크 테스트
  python exp5_stride4_survival.py --no-panels     # 시각화 패널 생략

출력 (HERE 기준):
  exp5_instances.csv.gz     인스턴스별 원자료
  exp5_survival_summary.md  크기구간/클래스/세장형별 생존율·복원IoU + 컷오프 정책표
  exp5_panels/*.png         대표 사례 시각화 (원본 crop | 1280 마스크 | 320 GT 복원)
"""
import argparse
import csv
import glob
import gzip
import importlib.util
import json
import math
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRAP_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RAW_DIRS = {
    "train": os.path.join(SCRAP_ROOT, "data", "datasets", "train_data"),
    "val": os.path.join(SCRAP_ROOT, "data", "datasets", "val_data"),
}
REMAP_SCRIPT = os.path.join(SCRAP_ROOT, "references", "pipeline", "2_remap_labelme_exact.py")
TARGET = 1280          # 학습 입력 (letterbox 긴 변)
RATIO = 4              # ultralytics mask_ratio (proto stride 4)
THIN_CLASSES = ("rebar", "pipe", "small pipe", "square pipe", "mesh")


def load_label_map():
    spec = importlib.util.spec_from_file_location("remap_mod", REMAP_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.LABEL_MAP


def shoelace(points):
    a = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0


def measure_instance(pts1280):
    """1280 좌표 폴리곤 1개 → (px1280, px320, iou, ncomp, px320_area, local rasters).

    ultralytics polygon2mask 재현: fillPoly(uint8, color=1) 후 cv2.resize(1/4) —
    기본 INTER_LINEAR라 얇은 형상은 위상에 따라 소실된다. 전역 stride-4 그리드와
    위상을 맞추기 위해 로컬 캔버스 원점을 4의 배수로 내림한다.
    """
    xs = pts1280[:, 0]
    ys = pts1280[:, 1]
    ox = int(max(0, math.floor(xs.min() / RATIO) * RATIO))
    oy = int(max(0, math.floor(ys.min() / RATIO) * RATIO))
    w = int(math.ceil((xs.max() - ox) / RATIO) * RATIO) + RATIO
    h = int(math.ceil((ys.max() - oy) / RATIO) * RATIO) + RATIO
    w = max(w, RATIO)
    h = max(h, RATIO)
    m1280 = np.zeros((h, w), dtype=np.uint8)
    local = np.round(pts1280 - [ox, oy]).astype(np.int32)
    cv2.fillPoly(m1280, [local], color=1)
    m320 = cv2.resize(m1280, (w // RATIO, h // RATIO))              # ultralytics와 동일 (INTER_LINEAR)
    m320_area = cv2.resize(m1280, (w // RATIO, h // RATIO), interpolation=cv2.INTER_AREA)
    px1280 = int(m1280.sum())
    px320 = int(m320.sum())
    if px320:
        up = cv2.resize(m320, (w, h), interpolation=cv2.INTER_NEAREST)
        inter = int(np.logical_and(up, m1280).sum())
        union = int(np.logical_or(up, m1280).sum())
        iou = inter / union if union else 0.0
        ncomp = cv2.connectedComponents(m320)[0] - 1
    else:
        iou, ncomp = 0.0, 0
    return px1280, px320, iou, ncomp, int(m320_area.sum()), (m1280, m320, ox, oy)


def collect(args):
    label_map = load_label_map()
    rows = []
    for split in args.splits:
        files = sorted(glob.glob(os.path.join(RAW_DIRS[split], "*.json")))
        if args.limit:
            files = files[: args.limit]
        for i, jp in enumerate(files):
            d = json.load(open(jp, encoding="utf-8"))
            W, H = d["imageWidth"], d["imageHeight"]
            s = TARGET / max(W, H)
            stem = os.path.splitext(os.path.basename(jp))[0]
            for si, sh in enumerate(d.get("shapes", [])):
                if sh.get("shape_type", "polygon") != "polygon":
                    continue
                raw_label = sh.get("label", "")
                if "Cargo Area" in raw_label:
                    continue
                merged = label_map.get(raw_label)
                if merged is None:
                    continue
                pts = np.asarray(sh["points"], dtype=np.float64)
                if len(pts) < 3:
                    continue
                pts1280 = pts * s
                bw = float(pts1280[:, 0].max() - pts1280[:, 0].min())
                bh = float(pts1280[:, 1].max() - pts1280[:, 1].min())
                sa = math.sqrt(shoelace([tuple(p) for p in pts])) * s   # prepare_yolo_dataset.py와 동일 정의
                px1280, px320, iou, ncomp, px320_area, rasters = measure_instance(pts1280)
                long_side = max(bw, bh)
                rows.append({
                    "split": split, "stem": stem, "shape_idx": si, "cls": merged,
                    "sa1280": round(sa, 2), "long1280": round(long_side, 1),
                    "short1280": round(min(bw, bh), 1),
                    "meanw1280": round(px1280 / long_side, 2) if long_side > 0 else 0.0,
                    "meanw_sh1280": round((sa * sa) / long_side, 2) if long_side > 0 else 0.0,
                    "px1280": px1280, "px320": px320, "px320_area": px320_area,
                    "survived": int(px320 > 0), "iou": round(iou, 3), "ncomp": ncomp,
                })
            if (i + 1) % 200 == 0:
                print("  [{}] {}/{} images, {} instances".format(split, i + 1, len(files), len(rows)))
    return rows


def fmt_pct(n, d):
    return "{:.1f}%".format(100.0 * n / d) if d else "-"


def median(vals):
    if not vals:
        return float("nan")
    v = sorted(vals)
    return v[len(v) // 2]


def summarize(rows, out_md):
    lines = ["# exp5 — stride 4 GT 마스크 생존 분석 (YOLO26x-seg @1280, mask_ratio=4 재현)", ""]
    lines.append("- 전수 대상: {} 인스턴스 (train+val 원본 LabelMe, Cargo 제외, 19클래스 매핑)".format(len(rows)))
    lines.append("- 재현 파이프라인: fillPoly@1280 → cv2.resize(×1/4, INTER_LINEAR) — ultralytics polygon2mask와 동일")
    lines.append("- survived: 320그리드에 픽셀 ≥1 생존 / iou: 320마스크를 1280으로 복원(NEAREST) 시 원마스크와 IoU")
    lines.append("")

    # 1) sqrt(area)@1280 구간별
    bins = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 12), (12, 16), (16, 24), (24, 48), (48, 10 ** 9)]
    lines.append("## 1. 크기(sqrt 폴리곤면적 @1280)별 생존율·복원 IoU")
    lines.append("")
    lines.append("| sqrt(area)@1280 | 원본 환산 | 개수 | 생존율 | IoU 중앙값 | IoU≥0.5 비율 | 조각남(comp≥2) |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for lo, hi in bins:
        grp = [r for r in rows if lo <= r["sa1280"] < hi]
        if not grp:
            continue
        n = len(grp)
        surv = sum(r["survived"] for r in grp)
        ious = [r["iou"] for r in grp]
        hi_lbl = "{:g}".format(hi) if hi < 10 ** 8 else "~"
        lines.append("| {:g}~{} px | {:g}~{} px | {} | {} | {:.2f} | {} | {} |".format(
            lo, hi_lbl, lo * 3, "{:g}".format(hi * 3) if hi < 10 ** 8 else "~", n,
            fmt_pct(surv, n), median(ious), fmt_pct(sum(1 for v in ious if v >= 0.5), n),
            fmt_pct(sum(1 for r in grp if r["ncomp"] >= 2), n)))
    lines.append("")

    # 2) 세장형: 면적 미달(<10px)인데 긴변 예외(>=24px)로 살아나는 그룹의 실제 생존
    lines.append("## 2. 세장형 예외 대상 (sa<10, 긴변≥24px@1280 = 원본 72px) — 두께별 실제 생존")
    lines.append("")
    lines.append("두께 정의 2종: raster=fillPoly 픽셀면적/긴변(GT가 실제 보는 두께), shoelace=다각형면적/긴변(라벨링 규칙·prepare_yolo_dataset.py와 동일).")
    exc = [r for r in rows if r["sa1280"] < 10 and r["long1280"] >= 24]
    wbins = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 10 ** 9)]
    for key, label in (("meanw1280", "raster"), ("meanw_sh1280", "shoelace")):
        lines.append("")
        lines.append("| 평균두께({})@1280 | 원본 환산 | 개수 | 생존율 | IoU 중앙값 | 조각남 |".format(label))
        lines.append("|---|---|---:|---:|---:|---:|")
        for lo, hi in wbins:
            grp = [r for r in exc if lo <= r[key] < hi]
            if not grp:
                continue
            n = len(grp)
            ious = [r["iou"] for r in grp]
            lines.append("| {:g}~{} px | {:g}~{} px | {} | {} | {:.2f} | {} |".format(
                lo, "{:g}".format(hi) if hi < 10 ** 8 else "~", lo * 3,
                "{:g}".format(hi * 3) if hi < 10 ** 8 else "~", n,
                fmt_pct(sum(r["survived"] for r in grp), n), median(ious),
                fmt_pct(sum(1 for r in grp if r["ncomp"] >= 2), n)))
    lines.append("")
    lines.append("(세장형 예외 전체 {}개 — 회신초안의 '약 2.9만 개' 검증)".format(len(exc)))
    lines.append("")

    # 3) 주요 얇은 클래스별
    lines.append("## 3. 세장형 클래스별 생존율 (전체 크기)")
    lines.append("")
    lines.append("| 클래스 | 개수 | 생존율 | IoU 중앙값 | IoU≥0.5 비율 |")
    lines.append("|---|---:|---:|---:|---:|")
    for cls in THIN_CLASSES:
        grp = [r for r in rows if r["cls"] == cls]
        if not grp:
            continue
        n = len(grp)
        ious = [r["iou"] for r in grp]
        lines.append("| {} | {} | {} | {:.2f} | {} |".format(
            cls, n, fmt_pct(sum(r["survived"] for r in grp), n), median(ious),
            fmt_pct(sum(1 for v in ious if v >= 0.5), n)))
    lines.append("")

    # 4) 컷오프 정책 시뮬레이션
    lines.append("## 4. 컷오프 정책별 라벨링 대상/신호 손실 시뮬레이션")
    lines.append("")
    lines.append("정책: A=면적만 / B=면적+긴변예외(현행) / C2·C3=긴변예외에 shoelace 두께≥2·3px@1280 추가")
    lines.append("(두께는 prepare_yolo_dataset.py --min-elongated-width와 동일한 shoelace 정의)")
    lines.append("")
    lines.append("| 컷 | 정책 | 유지 | 유지 중 GT소실(헛라벨) | 유지 중 IoU<0.5 | 제거 중 GT생존(신호손실) |")
    lines.append("|---:|---|---:|---:|---:|---:|")
    total = len(rows)
    for cut in (8, 10, 12, 16):
        for pol in ("A", "B", "C2", "C3"):
            wth = {"C2": 2, "C3": 3}.get(pol, 0)
            kept, cut_rows = [], []
            for r in rows:
                if r["sa1280"] >= cut:
                    kept.append(r)
                elif pol == "B" and r["long1280"] >= 24:
                    kept.append(r)
                elif wth and r["long1280"] >= 24 and r["meanw_sh1280"] >= wth:
                    kept.append(r)
                else:
                    cut_rows.append(r)
            nk = len(kept)
            lines.append("| {}px | {} | {} ({}) | {} | {} | {} |".format(
                cut, pol, nk, fmt_pct(nk, total),
                fmt_pct(sum(1 for r in kept if not r["survived"]), nk),
                fmt_pct(sum(1 for r in kept if r["iou"] < 0.5), nk),
                fmt_pct(sum(1 for r in cut_rows if r["survived"]), len(cut_rows))))
    lines.append("")

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("summary ->", out_md)


def render_panels(rows, args):
    """대표 사례 패널: 원본 crop + 1280 마스크 + 320 GT 복원(NEAREST 업샘플)."""
    out_dir = os.path.join(HERE, "exp5_panels")
    os.makedirs(out_dir, exist_ok=True)
    picks = []
    exc_dead = sorted((r for r in rows if r["sa1280"] < 10 and r["long1280"] >= 24
                       and r["cls"] in THIN_CLASSES and (not r["survived"] or r["iou"] < 0.2)),
                      key=lambda r: -r["long1280"])[:4]
    exc_live = sorted((r for r in rows if r["sa1280"] < 10 and r["long1280"] >= 24
                       and r["meanw1280"] >= 3 and r["iou"] >= 0.5), key=lambda r: -r["long1280"])[:2]
    compact = [r for r in rows if 9 <= r["sa1280"] <= 11 and r["long1280"] < 24 and r["iou"] >= 0.5][:2]
    picks = exc_dead + exc_live + compact
    label_map = load_label_map()
    for k, r in enumerate(picks):
        jp = os.path.join(RAW_DIRS[r["split"]], r["stem"] + ".json")
        ip = os.path.join(RAW_DIRS[r["split"]], r["stem"] + ".jpg")
        if not os.path.exists(ip):
            continue
        d = json.load(open(jp, encoding="utf-8"))
        sh = d["shapes"][r["shape_idx"]]
        if label_map.get(sh.get("label", "")) != r["cls"]:   # 인덱스 어긋남 방어
            continue
        W, H = d["imageWidth"], d["imageHeight"]
        s = TARGET / max(W, H)
        pts1280 = np.asarray(sh["points"], dtype=np.float64) * s
        _, _, _, _, _, (m1280, m320, ox, oy) = measure_instance(pts1280)
        h, w = m1280.shape
        img = cv2.imread(ip)
        img1280 = cv2.resize(img, (round(W * s), round(H * s)))
        crop = img1280[oy:oy + h, ox:ox + w]
        ch, cw = crop.shape[:2]
        if ch < h or cw < w:   # 이미지 경계 패딩
            crop = cv2.copyMakeBorder(crop, 0, h - ch, 0, w - cw, cv2.BORDER_CONSTANT)
        up = cv2.resize(m320, (w, h), interpolation=cv2.INTER_NEAREST)
        def colorize(mask, base):
            vis = base.copy()
            vis[mask > 0] = (0.4 * vis[mask > 0] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
            return vis
        scale_up = max(1, 320 // max(h, 1))
        panel = np.concatenate([colorize(m1280, crop), colorize(up, crop)], axis=1)
        panel = cv2.resize(panel, (panel.shape[1] * scale_up, panel.shape[0] * scale_up),
                           interpolation=cv2.INTER_NEAREST)
        name = "panel{:02d}_{}_{}_sa{:g}_w{:g}_iou{:g}.png".format(
            k, r["cls"].replace(" ", ""), r["stem"][:12], r["sa1280"], r["meanw1280"], r["iou"])
        cv2.imwrite(os.path.join(out_dir, name), panel)
    print("panels ->", out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--limit", type=int, default=0, help="split당 이미지 수 제한 (0=전수)")
    ap.add_argument("--no-panels", action="store_true")
    ap.add_argument("--scale-factor", type=float, default=1.0,
                    help="증강 최악조건 시뮬레이션용 추가 축소 (예: 0.5 = mosaic/scale 하한)")
    args = ap.parse_args()

    global TARGET
    TARGET = int(TARGET * args.scale_factor)
    suffix = "" if args.scale_factor == 1.0 else "_s{:g}".format(args.scale_factor)
    rows = collect(args)
    csv_path = os.path.join(HERE, "exp5_instances{}.csv.gz".format(suffix))
    with gzip.open(csv_path, "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("csv ->", csv_path)
    summarize(rows, os.path.join(HERE, "exp5_survival_summary{}.md".format(suffix)))
    if not args.no_panels:
        render_panels(rows, args)


if __name__ == "__main__":
    main()
