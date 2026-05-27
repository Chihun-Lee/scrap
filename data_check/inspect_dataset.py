"""
Steel Scrap Dataset Inspector
=============================
원본 LabelMe JSON + 이미지를 분석하여:
1. 기본 통계 (이미지/라벨 수, 해상도, 인스턴스 수 등)
2. 클래스별 분포 (raw label → merged class 매핑 포함)
3. 객체 크기 분포 (bbox area, polygon area)
4. 이미지당 인스턴스 수 분포
5. 샘플 시각화 (이미지 + 라벨 오버레이)
6. 라벨 품질 점검 (빈 라벨, 매핑 안 되는 클래스, 작은 객체 등)

Usage:
    conda activate scrap
    python data_check/inspect_dataset.py
"""

import json
import os
import sys
from pathlib import Path
from collections import Counter, defaultdict

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import matplotlib.patches as mpatches


# ── 설정 ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "datasets" / "train_data"
VAL_DIR = ROOT / "datasets" / "val_data"
OUTPUT_DIR = ROOT / "data_check" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 89 raw → 19 merged 매핑 (2_remap_labelme_exact.py 기준)
LABEL_MAP = {
    # beam
    "18.Cut H-beam scrap": "beam",
    "21.h-beam, i-beam, and bar- shaped steel": "beam",
    "35.Base plates, I-beams, H-beams": "beam",
    # drum
    "13.Sealed drum": "drum",
    "52.Sealed metal container": "drum",
    "54.Roll-type sealed container": "drum",
    # Fan
    "84. Unknown-Fan cover": "Fan",
    # Guillotine
    "67. Guillotine": "Guillotine",
    # handler
    "74. Handler": "handler",
    # heavy iron
    "23.Mixed heavy iron scrap": "heavy iron",
    "60.Boiler tank": "heavy iron",
    "68. Magnet": "heavy iron",
    "76. Streetlight pole": "heavy iron",
    # LPG GAS cylinder
    "69. LPG GAS cylinder": "LPG GAS cylinder",
    # machine
    "11.Shredder": "machine",
    "14.Scrap automotive parts": "machine",
    "43.Textile machinery": "machine",
    "44.Mold machinery": "machine",
    "53.Gearbox": "machine",
    "61.Reducer": "machine",
    "64.Automotive Engine Parts": "machine",
    "65.Loom for printing": "machine",
    "66.pressed car side door": "machine",
    "89. Unknown-Machine": "machine",
    # mesh
    "24.Grating manhole cover": "mesh",
    "58.Rockfall protection net": "mesh",
    "63.Steel grating": "mesh",
    "83. Unknown-Manhole Cover": "mesh",
    # mixed steel
    "1.Laser cutting (thick plate)": "mixed steel",
    "10.Shredded general ferrous scrap": "mixed steel",
    "12.Worksite oxidized scrap": "mixed steel",
    "19.Forklift truck": "mixed steel",
    "22.Spring": "mixed steel",
    "25.Rebar coil scrap": "mixed steel",
    "38.Nail scrap": "mixed steel",
    "45.Shredded nails": "mixed steel",
    "55.Sorting Scrap Metal": "mixed steel",
    "56.Incinerated scrap metal": "mixed steel",
    "62.Rusty Chain": "mixed steel",
    "87. Unknown-Mobile stand sign": "mixed steel",
    # panel
    "15.Gangform": "panel",
    "36.Air duct": "panel",
    "5.Elevator door": "panel",
    "51.Color-coated steel plate": "panel",
    "57.Deck reinforcement steel": "panel",
    "59.Fireproof door leaf": "panel",
    "6.Panels": "panel",
    "7.Incorner (form)": "panel",
    "77. Paint Can Lid": "panel",
    # pipe
    "2.Pipe_1": "pipe",
    "20.Galvanized steel pipe": "pipe",
    "26.Scaffolding pipe": "pipe",
    "33.Water supply pipe": "pipe",
    "37.Black steel pipe": "pipe",
    "40.Scaffolding pipe-Scaffolding platform": "pipe",
    "9.Housepipe": "pipe",
    # plastic
    "75. Plastic": "plastic",
    # rebar
    "28.Formwork tie pin": "rebar",
    "32.Rebar wire": "rebar",
    "41.Coiled reinforcing bar": "rebar",
    "42.Steel wire": "rebar",
    "48.Thick scrap wire": "rebar",
    "78. Unknown-Rebar": "rebar",
    # small pipe
    "3.Pipe_2": "small pipe",
    "46.Lead pipe (copper pipe)": "small pipe",
    # square pipe
    "31.Square steel pipe": "square pipe",
    # structure steel
    "27.Scaffold base plate": "structure steel",
    "29.Structural steel shapes": "structure steel",
    "34.Clean sheet steel": "structure steel",
    "70. Cabinet": "structure steel",
    "71. Paint_Can": "structure steel",
    "79. Unknown-Panel": "structure steel",
    "80. Unknown-Square Pipe": "structure steel",
    "86. Unknown-Cabinet": "structure steel",
    # trash
    "16.Chair": "trash",
    "4.Ton Bag": "trash",
    "72. Unknown": "trash",
    "82. Unknown-Plastic": "trash",
    "85. Unknown-Sorting Scrap Metal": "trash",
    "88. Unknown-Spray paint cans": "trash",
    # vehicle
    "30.End-of-life vehicle scrap": "vehicle",
    "47.End-of-life vehicle shell": "vehicle",
    "8.Electronic devices": "vehicle",
    "81. Unknown-Vehicle Part": "vehicle",
    # Cargo Area (제외)
    "Cargo Area": "__cargo__",
}

MERGED_CLASSES = [
    "handler", "rebar", "structure steel", "mixed steel", "heavy iron",
    "panel", "square pipe", "mesh", "small pipe", "trash",
    "vehicle", "pipe", "plastic", "machine", "LPG GAS cylinder",
    "beam", "drum", "Fan", "Guillotine",
]

# 19 클래스 색상
np.random.seed(42)
CLASS_COLORS = {c: tuple(np.random.randint(50, 255, 3).tolist()) for c in MERGED_CLASSES}


def load_labelme_json(json_path):
    """LabelMe JSON 파일 파싱"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def polygon_area(points):
    """Shoelace formula로 polygon 면적 계산"""
    pts = np.array(points)
    if len(pts) < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def polygon_bbox(points):
    """polygon → bounding box (x_min, y_min, x_max, y_max)"""
    pts = np.array(points)
    return pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()


def analyze_split(data_dir, split_name):
    """한 split(train/val) 전체 분석"""
    json_files = sorted(data_dir.glob("*.json"))
    img_files = sorted(data_dir.glob("*.jpg"))

    print(f"\n{'='*60}")
    print(f"  {split_name.upper()} Split 분석")
    print(f"{'='*60}")
    print(f"  이미지 수: {len(img_files)}")
    print(f"  JSON 수:   {len(json_files)}")

    # 이미지-JSON 매칭 확인
    img_stems = {f.stem for f in img_files}
    json_stems = {f.stem for f in json_files}
    img_only = img_stems - json_stems
    json_only = json_stems - img_stems
    if img_only:
        print(f"  ⚠ JSON 없는 이미지: {len(img_only)}개 — {list(img_only)[:5]}")
    if json_only:
        print(f"  ⚠ 이미지 없는 JSON: {len(json_only)}개 — {list(json_only)[:5]}")

    # 통계 수집
    raw_label_counter = Counter()
    merged_label_counter = Counter()
    unmapped_labels = Counter()
    cargo_count = 0
    instances_per_image = []
    poly_areas = []
    bbox_areas = []
    bbox_sizes = []  # (w, h)
    img_sizes = []
    empty_label_files = []
    small_objects = []  # polygon area < threshold
    per_image_class_dist = []  # 이미지별 클래스 구성

    for jf in json_files:
        data = load_labelme_json(jf)
        img_w = data.get("imageWidth", 3840)
        img_h = data.get("imageHeight", 2160)
        img_sizes.append((img_w, img_h))

        shapes = data.get("shapes", [])
        valid_count = 0
        img_classes = []

        for shape in shapes:
            raw_label = shape["label"]
            raw_label_counter[raw_label] += 1

            merged = LABEL_MAP.get(raw_label)
            if merged is None:
                # 숫자 prefix 제거 후 재시도
                stripped = raw_label.split(". ", 1)[-1] if ". " in raw_label else raw_label
                merged = LABEL_MAP.get(stripped)

            if merged is None:
                unmapped_labels[raw_label] += 1
                continue
            if merged == "__cargo__":
                cargo_count += 1
                continue

            merged_label_counter[merged] += 1
            valid_count += 1
            img_classes.append(merged)

            pts = shape.get("points", [])
            if len(pts) >= 3:
                area = polygon_area(pts)
                poly_areas.append((merged, area))
                x0, y0, x1, y1 = polygon_bbox(pts)
                bw, bh = x1 - x0, y1 - y0
                bbox_areas.append((merged, bw * bh))
                bbox_sizes.append((merged, bw, bh))

                # YOLO 640 리사이즈 기준 작은 객체
                scale = 640 / max(img_w, img_h)
                if bw * scale < 8 or bh * scale < 8:
                    small_objects.append((jf.stem, raw_label, bw, bh, bw * scale, bh * scale))

        instances_per_image.append(valid_count)
        per_image_class_dist.append(Counter(img_classes))
        if valid_count == 0:
            empty_label_files.append(jf.stem)

    return {
        "split": split_name,
        "n_images": len(img_files),
        "n_jsons": len(json_files),
        "img_sizes": img_sizes,
        "raw_label_counter": raw_label_counter,
        "merged_label_counter": merged_label_counter,
        "unmapped_labels": unmapped_labels,
        "cargo_count": cargo_count,
        "instances_per_image": instances_per_image,
        "poly_areas": poly_areas,
        "bbox_areas": bbox_areas,
        "bbox_sizes": bbox_sizes,
        "empty_label_files": empty_label_files,
        "small_objects": small_objects,
        "per_image_class_dist": per_image_class_dist,
    }


def print_stats(stats):
    """정량적 통계 출력"""
    s = stats
    inst = s["instances_per_image"]

    print(f"\n── 이미지 해상도 ──")
    sizes = Counter(s["img_sizes"])
    for (w, h), cnt in sizes.most_common():
        print(f"  {w}×{h}: {cnt}장")

    print(f"\n── 인스턴스 통계 ──")
    print(f"  총 인스턴스: {sum(inst)}")
    print(f"  이미지당 평균: {np.mean(inst):.1f}")
    print(f"  이미지당 중앙값: {np.median(inst):.0f}")
    print(f"  이미지당 최소/최대: {min(inst)} / {max(inst)}")
    print(f"  Cargo Area 라벨 수: {s['cargo_count']}")

    print(f"\n── 클래스별 인스턴스 수 (merged 19 classes) ──")
    mc = s["merged_label_counter"]
    total = sum(mc.values())
    for cls in MERGED_CLASSES:
        cnt = mc.get(cls, 0)
        pct = cnt / total * 100 if total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {cls:20s} {cnt:6d} ({pct:5.1f}%) {bar}")

    if s["unmapped_labels"]:
        print(f"\n── ⚠ 매핑 안 되는 라벨 ({len(s['unmapped_labels'])}종) ──")
        for label, cnt in s["unmapped_labels"].most_common(20):
            print(f"  {label}: {cnt}")

    if s["empty_label_files"]:
        print(f"\n── ⚠ 인스턴스 0개인 이미지: {len(s['empty_label_files'])}장 ──")
        for name in s["empty_label_files"][:10]:
            print(f"  {name}")

    if s["small_objects"]:
        print(f"\n── ⚠ YOLO 640 기준 8px 미만 객체: {len(s['small_objects'])}개 ──")
        for name, label, bw, bh, sw, sh in s["small_objects"][:10]:
            print(f"  {name} | {label} | 원본 {bw:.0f}×{bh:.0f} → 640스케일 {sw:.1f}×{sh:.1f}")

    # bbox 면적 통계 (클래스별)
    print(f"\n── 객체 크기 통계 (bbox area, px²) ──")
    by_cls = defaultdict(list)
    for cls, area in s["bbox_areas"]:
        by_cls[cls].append(area)
    print(f"  {'클래스':20s} {'평균':>10s} {'중앙값':>10s} {'최소':>10s} {'최대':>10s}")
    for cls in MERGED_CLASSES:
        if cls in by_cls:
            arr = by_cls[cls]
            print(f"  {cls:20s} {np.mean(arr):10.0f} {np.median(arr):10.0f} {min(arr):10.0f} {max(arr):10.0f}")


def plot_class_distribution(train_stats, val_stats, output_dir):
    """클래스별 인스턴스 분포 비교 차트"""
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(MERGED_CLASSES))
    w = 0.35

    train_counts = [train_stats["merged_label_counter"].get(c, 0) for c in MERGED_CLASSES]
    val_counts = [val_stats["merged_label_counter"].get(c, 0) for c in MERGED_CLASSES]

    ax.bar(x - w/2, train_counts, w, label="Train", color="#4C72B0")
    ax.bar(x + w/2, val_counts, w, label="Val", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(MERGED_CLASSES, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Instance Count")
    ax.set_title("Class Distribution (Train vs Val)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = output_dir / "class_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  저장: {path}")


def plot_instances_per_image(train_stats, val_stats, output_dir):
    """이미지당 인스턴스 수 히스토그램"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, stats, color in zip(axes, [train_stats, val_stats], ["#4C72B0", "#DD8452"]):
        inst = stats["instances_per_image"]
        ax.hist(inst, bins=50, color=color, edgecolor="white", alpha=0.8)
        ax.axvline(np.mean(inst), color="red", linestyle="--", label=f"mean={np.mean(inst):.1f}")
        ax.axvline(np.median(inst), color="green", linestyle="--", label=f"median={np.median(inst):.0f}")
        ax.set_title(f"{stats['split'].upper()} — Instances per Image")
        ax.set_xlabel("Instance count")
        ax.set_ylabel("Image count")
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    path = output_dir / "instances_per_image.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  저장: {path}")


def plot_object_size_distribution(train_stats, output_dir):
    """객체 크기(bbox area) 분포 — 전체 + 클래스별"""
    # 전체 분포
    areas = [a for _, a in train_stats["bbox_areas"]]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(areas, bins=100, color="#4C72B0", edgecolor="white", alpha=0.8)
    ax.set_title("Object Size Distribution (bbox area, Train)")
    ax.set_xlabel("Bbox Area (px²)")
    ax.set_ylabel("Count")
    ax.set_xlim(0, np.percentile(areas, 99))
    ax.grid(alpha=0.3)

    # log scale
    ax = axes[1]
    log_areas = [np.log10(a + 1) for a in areas]
    ax.hist(log_areas, bins=80, color="#55A868", edgecolor="white", alpha=0.8)
    ax.set_title("Object Size Distribution (log10, Train)")
    ax.set_xlabel("log10(Bbox Area)")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = output_dir / "object_size_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  저장: {path}")


def plot_bbox_aspect_ratio(train_stats, output_dir):
    """객체 종횡비 분포"""
    ratios = []
    for cls, w, h in train_stats["bbox_sizes"]:
        if h > 0:
            ratios.append(w / h)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ratios, bins=80, range=(0, 5), color="#C44E52", edgecolor="white", alpha=0.8)
    ax.axvline(1.0, color="black", linestyle="--", alpha=0.5, label="1:1")
    ax.set_title("Bbox Aspect Ratio (W/H) Distribution")
    ax.set_xlabel("W / H")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = output_dir / "aspect_ratio.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  저장: {path}")


def plot_class_cooccurrence(train_stats, output_dir):
    """클래스 공존 히트맵 — 같은 이미지에 함께 등장하는 클래스"""
    n = len(MERGED_CLASSES)
    cooc = np.zeros((n, n), dtype=int)
    cls_to_idx = {c: i for i, c in enumerate(MERGED_CLASSES)}

    for img_dist in train_stats["per_image_class_dist"]:
        present = [cls_to_idx[c] for c in img_dist if c in cls_to_idx]
        for i in present:
            for j in present:
                cooc[i, j] += 1

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cooc, cmap="YlOrRd")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(MERGED_CLASSES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(MERGED_CLASSES, fontsize=8)
    ax.set_title("Class Co-occurrence Matrix (Train)")
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    path = output_dir / "class_cooccurrence.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  저장: {path}")


def visualize_samples(data_dir, split_name, output_dir, n_samples=8):
    """샘플 이미지 + 라벨 오버레이 시각화"""
    json_files = sorted(data_dir.glob("*.json"))
    # 인스턴스 수 다양하게 샘플링
    np.random.seed(0)
    indices = np.random.choice(len(json_files), min(n_samples, len(json_files)), replace=False)
    indices = sorted(indices)

    cols = 4
    rows = (len(indices) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if rows == 1:
        axes = [axes]
    axes = np.array(axes).flatten()

    for idx, ax in enumerate(axes):
        if idx >= len(indices):
            ax.axis("off")
            continue

        jf = json_files[indices[idx]]
        img_path = jf.with_suffix(".jpg")
        if not img_path.exists():
            ax.axis("off")
            continue

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        overlay = img.copy()

        data = load_labelme_json(jf)
        class_set = set()
        for shape in data.get("shapes", []):
            raw_label = shape["label"]
            merged = LABEL_MAP.get(raw_label)
            if merged is None:
                stripped = raw_label.split(". ", 1)[-1] if ". " in raw_label else raw_label
                merged = LABEL_MAP.get(stripped)
            if merged is None or merged == "__cargo__":
                continue

            class_set.add(merged)
            pts = np.array(shape["points"], dtype=np.int32)
            color = CLASS_COLORS.get(merged, (200, 200, 200))
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(overlay, [pts], True, color, 2)

        blended = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)
        # 축소
        h, w = blended.shape[:2]
        scale = 640 / max(h, w)
        blended = cv2.resize(blended, (int(w * scale), int(h * scale)))

        ax.imshow(blended)
        n_inst = sum(1 for s in data.get("shapes", [])
                     if LABEL_MAP.get(s["label"], LABEL_MAP.get(s["label"].split(". ", 1)[-1] if ". " in s["label"] else s["label"])) not in (None, "__cargo__"))
        ax.set_title(f"{jf.stem}\n{n_inst} inst, {len(class_set)} cls", fontsize=9)
        ax.axis("off")

    plt.suptitle(f"{split_name.upper()} — Sample Visualizations", fontsize=14)
    plt.tight_layout()
    path = output_dir / f"samples_{split_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  저장: {path}")


def plot_class_legend(output_dir):
    """클래스별 색상 범례"""
    fig, ax = plt.subplots(figsize=(6, 6))
    patches = []
    for cls in MERGED_CLASSES:
        color = [c / 255.0 for c in CLASS_COLORS[cls]]
        patches.append(mpatches.Patch(color=color, label=cls))
    ax.legend(handles=patches, loc="center", fontsize=11, ncol=1)
    ax.axis("off")
    ax.set_title("Class Color Legend", fontsize=14)
    plt.tight_layout()
    path = output_dir / "class_legend.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  저장: {path}")


def save_report(train_stats, val_stats, output_dir):
    """분석 결과 텍스트 리포트 저장"""
    path = output_dir / "report.txt"
    import io
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf

    print("=" * 60)
    print("  Steel Scrap Dataset Analysis Report")
    print("=" * 60)

    for stats in [train_stats, val_stats]:
        print_stats(stats)

    # 전체 요약
    t_total = sum(train_stats["merged_label_counter"].values())
    v_total = sum(val_stats["merged_label_counter"].values())
    print(f"\n{'='*60}")
    print(f"  전체 요약")
    print(f"{'='*60}")
    print(f"  Train: {train_stats['n_images']}장, {t_total} 인스턴스")
    print(f"  Val:   {val_stats['n_images']}장, {v_total} 인스턴스")
    print(f"  Total: {train_stats['n_images'] + val_stats['n_images']}장, {t_total + v_total} 인스턴스")

    # 클래스 불균형 경고
    print(f"\n── 클래스 불균형 경고 ──")
    counts = [train_stats["merged_label_counter"].get(c, 0) for c in MERGED_CLASSES]
    if counts:
        max_c, min_c = max(counts), min(c for c in counts if c > 0) if any(c > 0 for c in counts) else 0
        ratio = max_c / min_c if min_c > 0 else float("inf")
        max_cls = MERGED_CLASSES[counts.index(max_c)]
        min_cls = MERGED_CLASSES[counts.index(min(c for c in counts if c > 0))] if min_c > 0 else "N/A"
        print(f"  최다: {max_cls} ({max_c})")
        print(f"  최소: {min_cls} ({min_c})")
        print(f"  불균형 비율: {ratio:.1f}:1")

    sys.stdout = old_stdout
    report = buf.getvalue()

    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  리포트 저장: {path}")

    # 콘솔에도 출력
    print(report)


def main():
    print("Steel Scrap Dataset Inspector")
    print(f"Train: {TRAIN_DIR}")
    print(f"Val:   {VAL_DIR}")
    print(f"Output: {OUTPUT_DIR}")

    if not TRAIN_DIR.exists():
        print(f"ERROR: {TRAIN_DIR} 가 없습니다. datasets.zip을 먼저 압축 해제하세요.")
        sys.exit(1)

    # 분석
    print("\n[1/8] Train 분석 중...")
    train_stats = analyze_split(TRAIN_DIR, "train")

    print("\n[2/8] Val 분석 중...")
    val_stats = analyze_split(VAL_DIR, "val")

    # 통계 출력 + 리포트 저장
    print("\n[3/8] 리포트 생성...")
    save_report(train_stats, val_stats, OUTPUT_DIR)

    # 차트
    print("\n[4/8] 클래스 분포 차트...")
    plot_class_distribution(train_stats, val_stats, OUTPUT_DIR)

    print("\n[5/8] 이미지당 인스턴스 분포...")
    plot_instances_per_image(train_stats, val_stats, OUTPUT_DIR)

    print("\n[6/8] 객체 크기 분포...")
    plot_object_size_distribution(train_stats, OUTPUT_DIR)
    plot_bbox_aspect_ratio(train_stats, OUTPUT_DIR)

    print("\n[7/8] 클래스 공존 히트맵...")
    plot_class_cooccurrence(train_stats, OUTPUT_DIR)

    print("\n[8/8] 샘플 시각화...")
    visualize_samples(TRAIN_DIR, "train", OUTPUT_DIR, n_samples=8)
    visualize_samples(VAL_DIR, "val", OUTPUT_DIR, n_samples=8)
    plot_class_legend(OUTPUT_DIR)

    print(f"\n✅ 분석 완료! 결과: {OUTPUT_DIR}/")
    print(f"   - report.txt              : 정량 분석 텍스트")
    print(f"   - class_distribution.png  : 클래스별 분포")
    print(f"   - instances_per_image.png : 이미지당 인스턴스 수")
    print(f"   - object_size_distribution.png : 객체 크기 분포")
    print(f"   - aspect_ratio.png        : 종횡비 분포")
    print(f"   - class_cooccurrence.png  : 클래스 공존 히트맵")
    print(f"   - samples_train.png       : Train 샘플 시각화")
    print(f"   - samples_val.png         : Val 샘플 시각화")
    print(f"   - class_legend.png        : 색상 범례")


if __name__ == "__main__":
    main()
