import json
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import defaultdict

from PIL import Image


# 면적 최소 기준 패치 버전

# =========================
# 설정
# =========================
YOLO_IMGSZ = 1024
MIN_SIZE_AFTER_RESIZE = 8.0
MIN_AREA_AFTER_RESIZE = 64.0

DATASET_PAIRS = [
    (
        Path("datasets/train_cropped"),
        Path("datasets/train_data_filtered"),
        Path("datasets/images/train"),
    ),
    (
        Path("datasets/val_cropped"),
        Path("datasets/val_data_filtered"),
        Path("datasets/images/val"),
    ),
]


# =========================
# 유틸
# =========================
def get_image_size(json_data: Dict[str, Any], json_path: Path) -> Tuple[int, int]:
    w = json_data.get("imageWidth")
    h = json_data.get("imageHeight")

    if isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0:
        return w, h

    candidates = []

    image_path_in_json = json_data.get("imagePath")
    if image_path_in_json:
        candidates.append((json_path.parent / image_path_in_json).resolve())

    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        candidates.append((json_path.parent / f"{json_path.stem}{ext}").resolve())

    for p in candidates:
        if p.exists():
            with Image.open(p) as im:
                return im.size

    raise FileNotFoundError(f"이미지 크기를 찾을 수 없습니다: {json_path}")


def polygon_bbox(points: List[List[float]]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def polygon_area(points: List[List[float]]) -> float:
    if not points or len(points) < 3:
        return 0.0

    area = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def resize_scale_for_yolo(img_w: int, img_h: int, imgsz: int = 640) -> float:
    return min(imgsz / img_w, imgsz / img_h)


def should_keep_shape(
    shape: Dict[str, Any],
    scale: float,
    min_size_after_resize: float,
    min_area_after_resize: float,
) -> bool:
    shape_type = shape.get("shape_type", "polygon")
    if shape_type != "polygon":
        return True

    points = shape.get("points", [])
    if not points or len(points) < 3:
        return False

    x1, y1, x2, y2 = polygon_bbox(points)
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)

    bw_r = bw * scale
    bh_r = bh * scale

    area = polygon_area(points)
    area_r = area * (scale ** 2)

    if min(bw_r, bh_r) < min_size_after_resize:
        return False

    if area_r < min_area_after_resize:
        return False

    return True


def find_image_for_json(json_path: Path, json_data: Dict[str, Any]) -> Path | None:
    image_path_in_json = json_data.get("imagePath")
    if image_path_in_json:
        p = (json_path.parent / image_path_in_json).resolve()
        if p.exists():
            return p

    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        p = (json_path.parent / f"{json_path.stem}{ext}").resolve()
        if p.exists():
            return p

    return None


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_images(input_dir: Path, image_ref_dir: Path) -> None:
    image_ref_dir.mkdir(parents=True, exist_ok=True)

    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        for img_path in input_dir.glob(f"*{ext}"):
            dst = image_ref_dir / img_path.name
            shutil.copy2(img_path, dst)


# =========================
# 핵심 처리
# =========================
def process_one_json(
    json_path: Path,
    output_dir: Path,
    image_ref_dir: Path,
    class_stats: Dict[str, Dict[str, int]],
) -> Tuple[int, int, int]:

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_w, img_h = get_image_size(data, json_path)
    scale = resize_scale_for_yolo(img_w, img_h, YOLO_IMGSZ)

    original_shapes = data.get("shapes", [])
    kept_shapes = []

    for shape in original_shapes:
        label = shape.get("label", "UNKNOWN")

        class_stats[label]["orig"] += 1

        if should_keep_shape(
            shape,
            scale,
            MIN_SIZE_AFTER_RESIZE,
            MIN_AREA_AFTER_RESIZE,
        ):
            kept_shapes.append(shape)
            class_stats[label]["kept"] += 1
        else:
            class_stats[label]["removed"] += 1

    data["shapes"] = kept_shapes

    img_path = find_image_for_json(json_path, data)
    if img_path is None or not img_path.exists():
        raise FileNotFoundError(f"이미지 없음: {json_path}")

    rel_image_path = Path("..") / "images" / image_ref_dir.name / img_path.name
    data["imagePath"] = rel_image_path.as_posix()

    output_dir.mkdir(parents=True, exist_ok=True)

    out_json = output_dir / json_path.name
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return len(original_shapes), len(kept_shapes), len(original_shapes) - len(kept_shapes)


def process_dataset(input_dir: Path, output_dir: Path, image_ref_dir: Path) -> None:
    if not input_dir.exists():
        return

    reset_dir(output_dir)
    reset_dir(image_ref_dir)
    copy_images(input_dir, image_ref_dir)

    json_files = sorted(input_dir.glob("*.json"))

    class_stats = defaultdict(lambda: {"orig": 0, "kept": 0, "removed": 0})

    total_orig = total_kept = total_removed = 0

    for json_path in json_files:
        orig, kept, removed = process_one_json(
            json_path, output_dir, image_ref_dir, class_stats
        )
        total_orig += orig
        total_kept += kept
        total_removed += removed

    print("\n===== CLASS SUMMARY =====")
    for cls, stat in sorted(class_stats.items()):
        print(
            f"{cls:20s} | "
            f"orig={stat['orig']:6d} "
            f"kept={stat['kept']:6d} "
            f"removed={stat['removed']:6d}"
        )

    print("\n===== TOTAL =====")
    print(f"orig={total_orig} kept={total_kept} removed={total_removed}")


def main():
    for input_dir, output_dir, image_ref_dir in DATASET_PAIRS:
        process_dataset(input_dir, output_dir, image_ref_dir)


if __name__ == "__main__":
    main()
