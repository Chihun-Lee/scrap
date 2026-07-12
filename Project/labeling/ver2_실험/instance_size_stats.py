# -*- coding: utf-8 -*-
"""철스크랩 데이터셋 인스턴스 크기 분포 분석 (CPU, 표준라이브러리만).

- LabelMe JSON(train_remapped/val_remapped) 순회
- polygon 면적(shoelace), bbox(w,h) — 원본 해상도 기준
- 1280 letterbox 스케일 s = 1280/max(W,H) 로 환산: sqrt(area)*s, max(w,h)*s
- 컷오프 스윕(제외 = 값 < 컷오프):
  sqrt(area)@1280 ∈ {4,6,8,10,12,16,20,24,32}px
  bbox max-side@1280 동일 스윕
  이미지 면적 대비 비율(%) ∈ {0.001, 0.005, 0.01, 0.05, 0.1}
- 클래스별 sqrt(area)@1280 p5/p25/p50
출력: instance_size_stats_train.csv (train 집계), instance_size_stats_summary.md (train/val 전체)
"""
import csv
import glob
import json
import math
import os
from collections import defaultdict

BASE = "/Users/chihun/Code/철스크랩/scrap/data/datasets"
OUT_DIR = "/Users/chihun/Code/철스크랩/scrap/Project/labeling/ver2_실험"
SPLITS = {
    "train": os.path.join(BASE, "train_remapped"),
    "val": os.path.join(BASE, "val_remapped"),
}
TARGET = 1280
SQRT_CUTS = [4, 6, 8, 10, 12, 16, 20, 24, 32]
RATIO_CUTS = [0.001, 0.005, 0.01, 0.05, 0.1]  # 이미지 면적 대비 %


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
    """선형 보간 percentile (sorted_vals는 정렬된 리스트, p는 0~100)."""
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
    """split 내 모든 인스턴스의 (label, sqrt_area@1280, maxside@1280, area_ratio%) 수집."""
    recs = []  # (label, sa1280, ms1280, ratio_pct)
    n_files = 0
    n_bad = 0
    for path in sorted(glob.glob(os.path.join(split_dir, "*.json"))):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        n_files += 1
        W = d.get("imageWidth")
        H = d.get("imageHeight")
        if not W or not H:
            n_bad += 1
            continue
        s = TARGET / max(W, H)
        img_area = W * H
        for sh in d.get("shapes", []):
            if sh.get("shape_type", "polygon") != "polygon":
                continue
            pts = sh.get("points", [])
            if len(pts) < 3:
                continue
            area = shoelace(pts)
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            w = max(xs) - min(xs)
            h = max(ys) - min(ys)
            sa1280 = math.sqrt(area) * s
            ms1280 = max(w, h) * s
            ratio = area / img_area * 100.0
            recs.append((sh["label"], sa1280, ms1280, ratio))
    return recs, n_files, n_bad


def aggregate(recs):
    """전체 + 클래스별 집계."""
    by_cls = defaultdict(list)
    for r in recs:
        by_cls[r[0]].append(r)
    out = {}
    for key, rows in [("__ALL__", recs)] + sorted(by_cls.items()):
        n = len(rows)
        sa = sorted(r[1] for r in rows)
        ms = sorted(r[2] for r in rows)
        ra = sorted(r[3] for r in rows)
        entry = {"n": n}
        entry["sqrt_cut"] = {c: sum(1 for v in sa if v < c) for c in SQRT_CUTS}
        entry["ms_cut"] = {c: sum(1 for v in ms if v < c) for c in SQRT_CUTS}
        entry["ratio_cut"] = {c: sum(1 for v in ra if v < c) for c in RATIO_CUTS}
        entry["p5"] = percentile(sa, 5)
        entry["p25"] = percentile(sa, 25)
        entry["p50"] = percentile(sa, 50)
        out[key] = entry
    return out


def fmt_pct(cnt, n):
    return "{} ({:.2f}%)".format(cnt, cnt / n * 100.0 if n else 0.0)


def write_csv(agg, path):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        header = ["class", "n_instances"]
        header += ["sqrtA<{}px_cnt".format(c) for c in SQRT_CUTS]
        header += ["sqrtA<{}px_pct".format(c) for c in SQRT_CUTS]
        header += ["maxside<{}px_cnt".format(c) for c in SQRT_CUTS]
        header += ["maxside<{}px_pct".format(c) for c in SQRT_CUTS]
        header += ["ratio<{}%_cnt".format(c) for c in RATIO_CUTS]
        header += ["ratio<{}%_pct".format(c) for c in RATIO_CUTS]
        header += ["sqrtA1280_p5", "sqrtA1280_p25", "sqrtA1280_p50"]
        w.writerow(header)
        for key in ["__ALL__"] + sorted(k for k in agg if k != "__ALL__"):
            e = agg[key]
            n = e["n"]
            row = ["ALL" if key == "__ALL__" else key, n]
            row += [e["sqrt_cut"][c] for c in SQRT_CUTS]
            row += [round(e["sqrt_cut"][c] / n * 100, 3) for c in SQRT_CUTS]
            row += [e["ms_cut"][c] for c in SQRT_CUTS]
            row += [round(e["ms_cut"][c] / n * 100, 3) for c in SQRT_CUTS]
            row += [e["ratio_cut"][c] for c in RATIO_CUTS]
            row += [round(e["ratio_cut"][c] / n * 100, 3) for c in RATIO_CUTS]
            row += [round(e["p5"], 2), round(e["p25"], 2), round(e["p50"], 2)]
            w.writerow(row)


def md_cut_table(agg_by_split, cut_key, cuts, unit):
    """전체(ALL) 기준 split×컷오프 표."""
    lines = ["| split | n | " + " | ".join("<{}{}".format(c, unit) for c in cuts) + " |"]
    lines.append("|---|---|" + "---|" * len(cuts))
    for split, agg in agg_by_split.items():
        e = agg["__ALL__"]
        n = e["n"]
        cells = [fmt_pct(e[cut_key][c], n) for c in cuts]
        lines.append("| {} | {} | {} |".format(split, n, " | ".join(cells)))
    return "\n".join(lines)


def md_class_table(agg, cut_key, cuts, unit):
    lines = ["| class | n | " + " | ".join("<{}{}".format(c, unit) for c in cuts) + " |"]
    lines.append("|---|---|" + "---|" * len(cuts))
    for key in sorted(k for k in agg if k != "__ALL__"):
        e = agg[key]
        n = e["n"]
        cells = [fmt_pct(e[cut_key][c], n) for c in cuts]
        lines.append("| {} | {} | {} |".format(key, n, " | ".join(cells)))
    return "\n".join(lines)


def main():
    agg_by_split = {}
    meta = {}
    for split, d in SPLITS.items():
        recs, n_files, n_bad = collect(d)
        agg_by_split[split] = aggregate(recs)
        meta[split] = (n_files, len(recs), n_bad, d)
        print("[{}] files={} instances={} (bad W/H files={})".format(split, n_files, len(recs), n_bad))

    write_csv(agg_by_split["train"], os.path.join(OUT_DIR, "instance_size_stats_train.csv"))

    L = []
    L.append("# 철스크랩 데이터셋 인스턴스 크기 분포 — 1280 letterbox 기준 컷오프 분석\n")
    L.append("- 데이터: `train_remapped`(LabelMe, 19클래스 리맵), `val_remapped`")
    for split in ["train", "val"]:
        nf, ni, nb, d = meta[split]
        L.append("- {}: {} 파일, {} 인스턴스 ({})".format(split, nf, ni, d))
    L.append("- 스케일: s = 1280 / max(W,H) (3840×2160 → s=0.3333, 즉 1280×720 유효해상도)")
    L.append("- 제외 기준: 값 < 컷오프 (strict)\n")

    L.append("## 1. 전체 — sqrt(area)@1280 컷오프별 제외 수(비율)\n")
    L.append(md_cut_table(agg_by_split, "sqrt_cut", SQRT_CUTS, "px"))
    L.append("\n## 2. 전체 — bbox max-side@1280 컷오프별 제외 수(비율)\n")
    L.append(md_cut_table(agg_by_split, "ms_cut", SQRT_CUTS, "px"))
    L.append("\n## 3. 전체 — 이미지 면적 대비 인스턴스 면적 비율(%) 컷오프별 제외 수(비율)\n")
    L.append(md_cut_table(agg_by_split, "ratio_cut", RATIO_CUTS, "%"))

    for split in ["train", "val"]:
        agg = agg_by_split[split]
        L.append("\n## 4-{}. {} 클래스별 sqrt(area)@1280 컷오프별 제외 수(비율)\n".format(
            "1" if split == "train" else "2", split))
        L.append(md_class_table(agg, "sqrt_cut", SQRT_CUTS, "px"))

    L.append("\n## 5. 클래스별 sqrt(area)@1280 분포 percentile (p5 / p25 / p50)\n")
    L.append("| class | train n | train p5 | train p25 | train p50 | val n | val p5 | val p25 | val p50 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    tr, va = agg_by_split["train"], agg_by_split["val"]
    classes = sorted(set(k for k in tr if k != "__ALL__") | set(k for k in va if k != "__ALL__"))
    for c in ["__ALL__"] + classes:
        name = "ALL" if c == "__ALL__" else c
        te = tr.get(c)
        ve = va.get(c)

        def cell(e):
            if e is None:
                return ["-", "-", "-", "-"]
            return [str(e["n"]), "{:.1f}".format(e["p5"]), "{:.1f}".format(e["p25"]), "{:.1f}".format(e["p50"])]
        L.append("| {} | {} | {} |".format(name, " | ".join(cell(te)), " | ".join(cell(ve))))

    with open(os.path.join(OUT_DIR, "instance_size_stats_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("done ->", OUT_DIR)


if __name__ == "__main__":
    main()
