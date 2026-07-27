import os
import json
import math
from copy import deepcopy
from typing import List, Dict, Any, Optional, Tuple
import shutil

from PIL import Image
from shapely.geometry import Polygon, box
from shapely.validation import make_valid


# =========================
# 사용자 설정값
# =========================
DATASET_PAIRS = [
    ("./datasets/train_data", "./datasets/train_cropped"),
    ("./datasets/val_data", "./datasets/val_cropped"),
]

CARGO_LABEL = "73. Cargo Area"
PADDING = 20

# centroid / intersection / both
INSIDE_MODE = "centroid"

# INSIDE_MODE가 intersection 또는 both일 때 사용
MIN_INTERSECTION_RATIO = 0.3

# crop 경계에 걸친 polygon을 잘라서 살릴지 여부
CLIP_POLYGONS = True

# 라벨 앞 숫자 제거
# 예: "41.Coiled reinforcing bar" -> "Coiled reinforcing bar"
STRIP_NUMERIC_PREFIX = False

# 내부 polygon이 하나도 없어도 crop 결과 저장할지 여부
SAVE_EMPTY = False
# =========================


IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
OUTPUT_DIR_NAMES_TO_CLEAR = {"train_cropped", "val_cropped"}


def clear_output_dir(output_dir_raw: str) -> None:
    output_dir = os.path.abspath(output_dir_raw)
    datasets_root = os.path.abspath("./datasets")
    output_name = os.path.basename(output_dir)

    if output_name not in OUTPUT_DIR_NAMES_TO_CLEAR:
        raise ValueError(f"Refusing to clear unexpected output directory: {output_dir}")

    try:
        is_inside_datasets = os.path.commonpath([datasets_root, output_dir]) == datasets_root
    except ValueError:
        is_inside_datasets = False

    if not is_inside_datasets:
        raise ValueError(f"Refusing to clear output directory outside datasets: {output_dir}")

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)
    print(f"[CLEANED] {output_dir}")


def clear_output_dirs(dataset_pairs: List[Tuple[str, str]]) -> None:
    seen = set()
    for _, output_dir in dataset_pairs:
        output_dir_abs = os.path.abspath(output_dir)
        if output_dir_abs in seen:
            continue
        seen.add(output_dir_abs)
        clear_output_dir(output_dir_abs)


def safe_polygon(points: List[List[float]]) -> Optional[Polygon]:
    if not points or len(points) < 3:
        return None
    try:
        poly = Polygon(points)
        if not poly.is_valid:
            poly = make_valid(poly)

        if poly.is_empty:
            return None

        if poly.geom_type == "Polygon":
            return poly if poly.area > 0 else None

        if poly.geom_type == "MultiPolygon":
            polys = [p for p in poly.geoms if p.area > 0]
            if not polys:
                return None
            return max(polys, key=lambda p: p.area)

        return None
    except Exception:
        return None


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(v, hi))


def normalize_label(label: str) -> str:
    if not isinstance(label, str):
        return ""
    s = label.strip()
    i = 0
    while i < len(s) and (s[i].isdigit() or s[i] in [".", " "]):
        i += 1
    return s[i:].strip() if i < len(s) else s


def find_target_shape(shapes: List[Dict[str, Any]], cargo_label: str) -> Optional[Dict[str, Any]]:
    best_shape = None
    best_area = -1.0

    for shape in shapes:
        if shape.get("shape_type", "polygon") != "polygon":
            continue
        if shape.get("label") != cargo_label:
            continue

        poly = safe_polygon(shape.get("points", []))
        if poly is None:
            continue

        if poly.area > best_area:
            best_area = poly.area
            best_shape = shape

    return best_shape


def build_crop_box(cargo_poly: Polygon, img_w: int, img_h: int, padding: int) -> Tuple[int, int, int, int]:
    minx, miny, maxx, maxy = cargo_poly.bounds

    left = clamp(int(math.floor(minx - padding)), 0, img_w)
    top = clamp(int(math.floor(miny - padding)), 0, img_h)
    right = clamp(int(math.ceil(maxx + padding)), 0, img_w)
    bottom = clamp(int(math.ceil(maxy + padding)), 0, img_h)

    if right <= left or bottom <= top:
        raise ValueError("Invalid crop box")
    return left, top, right, bottom


def polygon_inside_cargo(poly: Polygon, cargo_poly: Polygon, inside_mode: str, min_ratio: float) -> bool:
    if poly is None or poly.is_empty or poly.area <= 0:
        return False

    centroid_inside = cargo_poly.buffer(1e-6).contains(poly.centroid)
    inter_area = cargo_poly.intersection(poly).area
    ratio = inter_area / poly.area if poly.area > 0 else 0.0

    if inside_mode == "centroid":
        return centroid_inside
    elif inside_mode == "intersection":
        return ratio >= min_ratio
    elif inside_mode == "both":
        return centroid_inside and ratio >= min_ratio
    else:
        raise ValueError(f"Unknown inside_mode: {inside_mode}")


def clip_to_crop(poly: Polygon, crop_rect) -> Optional[Polygon]:
    inter = poly.intersection(crop_rect)
    if inter.is_empty:
        return None

    if inter.geom_type == "Polygon":
        return inter if inter.area > 0 else None

    if inter.geom_type == "MultiPolygon":
        polys = [p for p in inter.geoms if p.area > 0]
        if not polys:
            return None
        return max(polys, key=lambda p: p.area)

    return None


def shift_points(poly: Polygon, left: int, top: int) -> List[List[float]]:
    coords = list(poly.exterior.coords)[:-1]
    return [[round(x - left, 2), round(y - top, 2)] for x, y in coords]


def find_matching_image(json_path: str) -> Optional[str]:
    stem = os.path.splitext(json_path)[0]
    for ext in IMAGE_EXTS:
        cand = stem + ext
        if os.path.exists(cand):
            return cand
    return None


def process_one(
    json_path: str,
    image_path: str,
    out_json_path: str,
    out_image_path: str,
    cargo_label: str,
    padding: int,
    inside_mode: str,
    min_intersection_ratio: float,
    clip_polygons: bool,
    strip_numeric_prefix: bool,
    save_empty: bool,
) -> bool:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    shapes = data.get("shapes", [])
    if not shapes:
        return False

    cargo_shape = find_target_shape(shapes, cargo_label)
    if cargo_shape is None:
        return False

    cargo_poly = safe_polygon(cargo_shape["points"])
    if cargo_poly is None:
        return False

    img = Image.open(image_path)
    img_w, img_h = img.size

    left, top, right, bottom = build_crop_box(cargo_poly, img_w, img_h, padding)
    crop_rect = box(left, top, right, bottom)

    cropped = img.crop((left, top, right, bottom))
    crop_w, crop_h = cropped.size

    new_shapes = []

    for shape in shapes:
        label = shape.get("label", "")
        shape_type = shape.get("shape_type", "polygon")

        if shape_type != "polygon":
            continue

        if label == cargo_label:
            continue

        poly = safe_polygon(shape.get("points", []))
        if poly is None:
            continue

        if not polygon_inside_cargo(poly, cargo_poly, inside_mode, min_intersection_ratio):
            continue

        if clip_polygons:
            poly2 = clip_to_crop(poly, crop_rect)
            if poly2 is None:
                continue
        else:
            if not crop_rect.buffer(1e-6).contains(poly):
                continue
            poly2 = poly

        new_points = shift_points(poly2, left, top)
        if len(new_points) < 3:
            continue

        new_shape = deepcopy(shape)
        new_shape["points"] = new_points

        if strip_numeric_prefix:
            new_shape["label"] = normalize_label(label)

        new_shapes.append(new_shape)

    if (not new_shapes) and (not save_empty):
        return False

    os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
    os.makedirs(os.path.dirname(out_image_path), exist_ok=True)

    cropped.save(out_image_path)

    new_data = deepcopy(data)
    new_data["imagePath"] = os.path.basename(out_image_path)
    new_data["imageHeight"] = crop_h
    new_data["imageWidth"] = crop_w
    new_data["imageData"] = None
    new_data["shapes"] = new_shapes

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    return True


def collect_json_files(input_dir: str) -> List[str]:
    results = []
    for root, _, files in os.walk(input_dir):
        for name in files:
            if name.lower().endswith(".json"):
                results.append(os.path.join(root, name))
    return sorted(results)


def process_dataset(input_dir_raw: str, output_dir_raw: str) -> Dict[str, int]:
    input_dir = os.path.abspath(input_dir_raw)
    output_dir = os.path.abspath(output_dir_raw)

    print("\n=== DATASET ===")
    print("INPUT_DIR :", input_dir)
    print("OUTPUT_DIR:", output_dir)
    print("INPUT EXISTS :", os.path.exists(input_dir))
    print("OUTPUT PARENT EXISTS :", os.path.exists(os.path.dirname(output_dir) or output_dir))

    json_files = collect_json_files(input_dir)
    print("JSON COUNT:", len(json_files))
    cropped = 0
    crop_failed = 0
    skipped = 0
    no_image = 0

    for json_path in json_files:
        image_path = find_matching_image(json_path)
        if image_path is None:
            no_image += 1
            print(f"[NO IMAGE] {json_path}")
            continue

        rel_dir = os.path.relpath(os.path.dirname(json_path), input_dir)
        base = os.path.splitext(os.path.basename(json_path))[0]
        ext = os.path.splitext(image_path)[1].lower()

        out_dir = os.path.join(output_dir, rel_dir)
        out_json_path = os.path.join(out_dir, base + ".json")
        out_image_path = os.path.join(out_dir, base + ext)

        try:
            ok = process_one(
                json_path=json_path,
                image_path=image_path,
                out_json_path=out_json_path,
                out_image_path=out_image_path,
                cargo_label=CARGO_LABEL,
                padding=PADDING,
                inside_mode=INSIDE_MODE,
                min_intersection_ratio=MIN_INTERSECTION_RATIO,
                clip_polygons=CLIP_POLYGONS,
                strip_numeric_prefix=STRIP_NUMERIC_PREFIX,
                save_empty=SAVE_EMPTY,
            )
            if ok:
                cropped += 1
                print(f"[CROPPED] {json_path}")
            else:
                crop_failed += 1
                print(f"[CROP FAILED] {json_path}")
        except Exception as e:
            skipped += 1
            print(f"[ERROR] {json_path} -> {e}")

    print("\n=== DONE ===")
    print(f"total   : {len(json_files)}")
    print(f"cropped : {cropped}")
    print(f"failed  : {crop_failed}")
    print(f"skipped : {skipped}")
    print(f"no_image: {no_image}")

    return {
        "total": len(json_files),
        "cropped": cropped,
        "crop_failed": crop_failed,
        "skipped": skipped,
        "no_image": no_image,
    }


def main():
    totals = {"total": 0, "cropped": 0, "crop_failed": 0, "skipped": 0, "no_image": 0}

    clear_output_dirs(DATASET_PAIRS)

    for input_dir, output_dir in DATASET_PAIRS:
        stats = process_dataset(input_dir, output_dir)
        for key in totals:
            totals[key] += stats[key]

    print("\n=== ALL DONE ===")
    print(f"total   : {totals['total']}")
    print(f"cropped : {totals['cropped']}")
    print(f"failed  : {totals['crop_failed']}")
    print(f"skipped : {totals['skipped']}")
    print(f"no_image: {totals['no_image']}")

    if totals["crop_failed"] or totals["skipped"] or totals["no_image"]:
        raise RuntimeError(
            "Cargo crop failed for one or more files. "
            "No original files were copied as fallback."
        )


if __name__ == "__main__":
    main()
