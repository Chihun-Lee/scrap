# -*- coding: utf-8 -*-
"""원본(미필터) LabelMe 라벨 기준 인스턴스 크기 분포 분석 — 1280 letterbox 환산.

instance_size_stats.py 와 동일한 스윕이지만 train_data/val_data (89 raw 클래스,
ver1 8px@640 필터 미적용 전체 417K/84K 인스턴스)를 대상으로 함.
"Cargo Area" 라벨은 제외. 클래스별 표는 인스턴스 수 상위 25개만.
출력: instance_size_stats_raw_summary.md
"""
import glob
import json
import math
import os
from collections import defaultdict

BASE = "/Users/chihun/Code/철스크랩/scrap/data/datasets"
OUT_DIR = "/Users/chihun/Code/철스크랩/scrap/Project/labeling/ver2_실험"
SPLITS = {
    "train": os.path.join(BASE, "train_data"),
    "val": os.path.join(BASE, "val_data"),
}
TARGET = 1280
SQRT_CUTS = [4, 6, 8, 10, 12, 16, 20, 24, 32]
RATIO_CUTS = [0.001, 0.005, 0.01, 0.05, 0.1]  # 이미지 면적 대비 %
TOP_CLASSES = 25


def shoelace(points):
    n = len(points)
    if n < 3:
        return 0.0
    a = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0


def percentile(sorted_vals, p):
    n = len(sorted_vals)
    if n == 0:
        return float("nan")
    if n == 1:
        return sorted_vals[0]
    k = (n - 1) * p / 100.0
    f = math.floor(k)
    c = min(f + 1, n - 1)
    return sorted_vals[int(f)] + (sorted_vals[c] - sorted_vals[int(f)]) * (k - f)


def collect(split_dir):
    recs = []  # (label, sqrt_area@1280, maxside@1280, ratio_pct)
    n_files = 0
    for path in sorted(glob.glob(os.path.join(split_dir, "*.json"))):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        n_files += 1
        W = d.get("imageWidth")
        H = d.get("imageHeight")
        if not W or not H:
            continue
        s = TARGET / max(W, H)
        img_area = W * H
        for sh in d.get("shapes", []):
            if sh.get("shape_type", "polygon") != "polygon":
                continue
            if "Cargo Area" in sh.get("label", ""):
                continue
            pts = sh.get("points", [])
            if len(pts) < 3:
                continue
            area = shoelace(pts)
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            w = max(xs) - min(xs)
            h = max(ys) - min(ys)
            recs.append((sh["label"], math.sqrt(area) * s, max(w, h) * s,
                         area / img_area * 100.0, min(w, h) * s))
    return recs, n_files


def aggregate(recs):
    by_cls = defaultdict(list)
    for r in recs:
        by_cls[r[0]].append(r)
    out = {}
    for key, rows in [("__ALL__", recs)] + sorted(by_cls.items()):
        n = len(rows)
        sa = sorted(r[1] for r in rows)
        ms = sorted(r[2] for r in rows)
        ra = sorted(r[3] for r in rows)
        mn = sorted(r[4] for r in rows)
        entry = {"n": n}
        entry["sqrt_cut"] = {c: sum(1 for v in sa if v < c) for c in SQRT_CUTS}
        entry["ms_cut"] = {c: sum(1 for v in ms if v < c) for c in SQRT_CUTS}
        entry["ratio_cut"] = {c: sum(1 for v in ra if v < c) for c in RATIO_CUTS}
        entry["minside_cut"] = {c: sum(1 for v in mn if v < c) for c in SQRT_CUTS}
        entry["p5"] = percentile(sa, 5)
        entry["p25"] = percentile(sa, 25)
        entry["p50"] = percentile(sa, 50)
        out[key] = entry
    return out


def fmt_pct(cnt, n):
    return "{} ({:.2f}%)".format(cnt, cnt / n * 100.0 if n else 0.0)


def md_cut_table(agg_by_split, cut_key, cuts, unit):
    lines = ["| split | n | " + " | ".join("<{}{}".format(c, unit) for c in cuts) + " |"]
    lines.append("|---|---|" + "---|" * len(cuts))
    for split, agg in agg_by_split.items():
        e = agg["__ALL__"]
        n = e["n"]
        cells = [fmt_pct(e[cut_key][c], n) for c in cuts]
        lines.append("| {} | {} | {} |".format(split, n, " | ".join(cells)))
    return "\n".join(lines)


def main():
    agg_by_split = {}
    meta = {}
    for split, d in SPLITS.items():
        recs, n_files = collect(d)
        agg_by_split[split] = aggregate(recs)
        meta[split] = (n_files, len(recs), d)
        print("[{}] files={} instances={}".format(split, n_files, len(recs)))

    L = []
    L.append("# 원본(미필터) 라벨 기준 인스턴스 크기 분포 — 1280 letterbox 환산\n")
    L.append("- 데이터: `train_data`/`val_data` (89 raw 클래스 LabelMe, ver1 8px@640 필터 미적용, Cargo Area 제외)")
    for split in ["train", "val"]:
        nf, ni, d = meta[split]
        L.append("- {}: {} 파일, {} 인스턴스 ({})".format(split, nf, ni, d))
    L.append("- 스케일: s = 1280 / max(W,H) (3840×2160 → s=0.3333)")
    L.append("- 제외 기준: 값 < 컷오프\n")

    L.append("## 1. 전체 — sqrt(area)@1280 컷오프별 제외 수(비율)\n")
    L.append(md_cut_table(agg_by_split, "sqrt_cut", SQRT_CUTS, "px"))
    L.append("\n## 2. 전체 — bbox max-side@1280 컷오프별 제외 수(비율)\n")
    L.append(md_cut_table(agg_by_split, "ms_cut", SQRT_CUTS, "px"))
    L.append("\n## 3. 전체 — 이미지 면적 대비 인스턴스 면적 비율(%) 컷오프별 제외 수(비율)\n")
    L.append(md_cut_table(agg_by_split, "ratio_cut", RATIO_CUTS, "%"))
    L.append("\n## 3b. 전체 — bbox min-side(짧은 변)@1280 컷오프별 제외 수(비율) — 기존 ver1/스윕 필터와 동일 정의\n")
    L.append(md_cut_table(agg_by_split, "minside_cut", SQRT_CUTS, "px"))

    tr = agg_by_split["train"]
    top = sorted((k for k in tr if k != "__ALL__"), key=lambda k: -tr[k]["n"])[:TOP_CLASSES]
    L.append("\n## 4. train 인스턴스 수 상위 {}개 raw 클래스 — sqrt(area)@1280 percentile 및 주요 컷오프 제외율\n".format(TOP_CLASSES))
    L.append("| raw class | n | p5 | p25 | p50 | <6px | <8px | <10px | <12px |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for k in ["__ALL__"] + top:
        e = tr[k]
        n = e["n"]
        L.append("| {} | {} | {:.1f} | {:.1f} | {:.1f} | {} | {} | {} | {} |".format(
            "ALL" if k == "__ALL__" else k, n, e["p5"], e["p25"], e["p50"],
            fmt_pct(e["sqrt_cut"][6], n), fmt_pct(e["sqrt_cut"][8], n),
            fmt_pct(e["sqrt_cut"][10], n), fmt_pct(e["sqrt_cut"][12], n)))

    out_path = os.path.join(OUT_DIR, "instance_size_stats_raw_summary.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("done ->", out_path)


if __name__ == "__main__":
    main()
