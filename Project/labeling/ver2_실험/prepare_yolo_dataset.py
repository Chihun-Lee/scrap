# -*- coding: utf-8 -*-
"""원본 LabelMe(train_data/val_data) → 19클래스 YOLO seg 데이터셋 변형 생성기.

ver2 라벨링 기준 연구(2·3순위)용. ver1 필터(8px@640)를 거치지 않은 원본 라벨에서
직접 시작해, 1280 입력 환산 기준으로 소형 컷오프/세장형 예외/폴리곤 단순화를 적용한다.

  python prepare_yolo_dataset.py --out cut8 --min-sqrt-area 8
  python prepare_yolo_dataset.py --out cut8_rdp4 --min-sqrt-area 8 --rdp-eps 4
  python prepare_yolo_dataset.py --out base            # 무필터

- --min-sqrt-area N : 1280 letterbox 환산 sqrt(폴리곤 면적) < N px 인 인스턴스 제거 (0=끔)
- --keep-elongated M: 면적 미달이어도 bbox 긴 변 >= M px(@1280)이면 유지 (기본 24, 0=끔)
- --rdp-eps E       : Douglas-Peucker 단순화 epsilon E px(@1280) (0=끔). 128점 초과 시 eps를 늘려 재시도
- 이미지는 복사하지 않고 심볼릭 링크. 출력: OUT_ROOT/<out>/{images,labels}/{train,val} + data.yaml
"""
import argparse
import glob
import importlib.util
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRAP_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))  # ~/Code/철스크랩/scrap
RAW_DIRS = {
    "train": os.path.join(SCRAP_ROOT, "data", "datasets", "train_data"),
    "val": os.path.join(SCRAP_ROOT, "data", "datasets", "val_data"),
}
CLASSES_TXT = os.path.join(SCRAP_ROOT, "data", "datasets", "classes.txt")
REMAP_SCRIPT = os.path.join(SCRAP_ROOT, "references", "pipeline", "2_remap_labelme_exact.py")
OUT_ROOT = os.path.join(HERE, "datasets_exp")
TARGET = 1280
MAX_POINTS = 128


def load_label_map():
    """2_remap_labelme_exact.py의 LABEL_MAP(89 raw → 19 merged) 재사용."""
    spec = importlib.util.spec_from_file_location("remap_mod", REMAP_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.LABEL_MAP


def rdp(points, eps):
    """Ramer-Douglas-Peucker (원본 좌표계, eps는 원본 px)."""
    if len(points) < 3 or eps <= 0:
        return points
    (x1, y1), (x2, y2) = points[0], points[-1]
    dx, dy = x2 - x1, y2 - y1
    norm = math.hypot(dx, dy)
    dmax, idx = -1.0, 0
    for i in range(1, len(points) - 1):
        px, py = points[i]
        d = abs(dy * px - dx * py + x2 * y1 - y2 * x1) / norm if norm > 0 else math.hypot(px - x1, py - y1)
        if d > dmax:
            dmax, idx = d, i
    if dmax > eps:
        left = rdp(points[: idx + 1], eps)
        right = rdp(points[idx:], eps)
        return left[:-1] + right
    return [points[0], points[-1]]


def simplify_polygon(pts, eps_orig):
    """닫힌 폴리곤 단순화 + 128점 상한 (초과 시 eps 1.5배씩 증가)."""
    if eps_orig <= 0 and len(pts) <= MAX_POINTS:
        return pts
    eps = max(eps_orig, 0.1)
    out = pts
    for _ in range(8):
        out = rdp(pts + [pts[0]], eps)[:-1]  # 시작점을 끝에 붙여 닫고, 결과에서 중복 제거
        if len(out) <= MAX_POINTS:
            break
        eps *= 1.5
    return out if len(out) >= 3 else pts


def shoelace(points):
    a = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="변형 이름 (datasets_exp/<out>/)")
    ap.add_argument("--min-sqrt-area", type=float, default=0.0, help="sqrt(area)@1280 컷오프 px")
    ap.add_argument("--keep-elongated", type=float, default=24.0, help="bbox 긴변@1280 유지 기준 px (0=끔)")
    ap.add_argument("--rdp-eps", type=float, default=0.0, help="RDP epsilon px(@1280)")
    args = ap.parse_args()

    label_map = load_label_map()
    classes = [c.strip() for c in open(CLASSES_TXT, encoding="utf-8") if c.strip()]
    cls_idx = {c: i for i, c in enumerate(classes)}

    out_dir = os.path.join(OUT_ROOT, args.out)
    stats = {"kept": 0, "cut_small": 0, "kept_elongated": 0, "dropped_unmapped": 0}
    for split, raw_dir in RAW_DIRS.items():
        img_out = os.path.join(out_dir, "images", split)
        lbl_out = os.path.join(out_dir, "labels", split)
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)
        for jp in sorted(glob.glob(os.path.join(raw_dir, "*.json"))):
            d = json.load(open(jp, encoding="utf-8"))
            W, H = d["imageWidth"], d["imageHeight"]
            s = TARGET / max(W, H)
            lines = []
            for sh in d.get("shapes", []):
                if sh.get("shape_type", "polygon") != "polygon":
                    continue
                raw_label = sh.get("label", "")
                if "Cargo Area" in raw_label:
                    continue
                merged = label_map.get(raw_label)
                if merged is None or merged not in cls_idx:
                    stats["dropped_unmapped"] += 1
                    continue
                pts = [(float(p[0]), float(p[1])) for p in sh["points"]]
                if len(pts) < 3:
                    continue
                if args.min_sqrt_area > 0:
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    sa = math.sqrt(shoelace(pts)) * s
                    ms = max(max(xs) - min(xs), max(ys) - min(ys)) * s
                    if sa < args.min_sqrt_area:
                        if args.keep_elongated > 0 and ms >= args.keep_elongated:
                            stats["kept_elongated"] += 1
                        else:
                            stats["cut_small"] += 1
                            continue
                if args.rdp_eps > 0:
                    pts = simplify_polygon(pts, args.rdp_eps / s)  # eps를 원본 px로 환산
                coords = []
                for x, y in pts:
                    coords.append(min(max(x / W, 0.0), 1.0))
                    coords.append(min(max(y / H, 0.0), 1.0))
                lines.append(str(cls_idx[merged]) + " " + " ".join("{:.6f}".format(v) for v in coords))
                stats["kept"] += 1
            stem = os.path.splitext(os.path.basename(jp))[0]
            with open(os.path.join(lbl_out, stem + ".txt"), "w") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
            src_img = os.path.join(raw_dir, stem + ".jpg")
            dst_img = os.path.join(img_out, stem + ".jpg")
            if os.path.exists(src_img) and not os.path.lexists(dst_img):
                os.symlink(src_img, dst_img)

    with open(os.path.join(out_dir, "data.yaml"), "w", encoding="utf-8") as f:
        f.write("path: {}\ntrain: images/train\nval: images/val\nnc: {}\nnames:\n".format(out_dir, len(classes)))
        for c in classes:
            f.write("  - {}\n".format(c))
    print("[{}] {}".format(args.out, stats))
    print("data.yaml ->", os.path.join(out_dir, "data.yaml"))


if __name__ == "__main__":
    sys.setrecursionlimit(100000)  # RDP 재귀 (폴리곤 점 수천 개 대비)
    main()
