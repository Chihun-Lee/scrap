import os
import json
import math
import glob
from datetime import date

import cv2
import numpy as np
from PIL import Image


def get_coco_data_anylabeling():
    # AnyLabeling get_coco_data("polygon") 스타일
    return {
        "info": {
            "year": date.today().year,
            "version": "0.4.16",
            "description": "COCO Label Conversion",
            "contributor": "CVHub",
            "url": "https://github.com/CVHub520/X-AnyLabeling",
            "date_created": str(date.today()),
        },
        "licenses": [
            {
                "id": 1,
                "url": "https://www.gnu.org/licenses/gpl-3.0.html",
                "name": "GNU GENERAL PUBLIC LICENSE Version 3",
            }
        ],
        "categories": [],
        "images": [],
        "annotations": [],
        "type": "instances",
    }


def get_image_size(image_file):
    with Image.open(image_file) as img:
        width, height = img.size
    return width, height


def find_image_for_json(input_dir, json_file, data):
    # 1) imagePath 우선
    image_path = data.get("imagePath")
    if image_path:
        candidate = os.path.join(input_dir, image_path)
        if os.path.exists(candidate):
            return candidate

    # 2) json basename 기준 탐색
    base = os.path.splitext(os.path.basename(json_file))[0]
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP"):
        candidate = os.path.join(input_dir, base + ext)
        if os.path.exists(candidate):
            return candidate

    # 3) 혹시 같은 prefix 파일 찾기
    candidates = glob.glob(os.path.join(input_dir, base + ".*"))
    for candidate in candidates:
        if candidate.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            return candidate

    return None


def points_to_segmentation(points):
    seg = []
    for x, y in points:
        seg.extend([float(x), float(y)])
    return [seg]


def get_min_enclosing_bbox_anylabeling(segmentations):
    """
    AnyLabeling get_min_enclosing_bbox 방식
    """
    all_polygon_points = []

    if not segmentations:
        return []

    for segmentation in segmentations:
        if not segmentation:
            continue

        polygon_points = [
            (segmentation[i], segmentation[i + 1])
            for i in range(0, len(segmentation), 2)
        ]
        all_polygon_points.extend(polygon_points)

    if not all_polygon_points:
        return []

    x_coords, y_coords = zip(*all_polygon_points)
    x_min_fp = min(x_coords)
    y_min_fp = min(y_coords)
    x_max_fp = max(x_coords)
    y_max_fp = max(y_coords)

    x_min_int = math.floor(x_min_fp)
    y_min_int = math.floor(y_min_fp)
    x_max_int = math.floor(x_max_fp)
    y_max_int = math.floor(y_max_fp)

    bbox_width = float(x_max_int - x_min_int + 1)
    bbox_height = float(y_max_int - y_min_int + 1)

    return [float(x_min_int), float(y_min_int), bbox_width, bbox_height]


def calculate_polygon_area_anylabeling(segmentations):
    """
    AnyLabeling calculate_polygon_area 방식
    """
    if not segmentations:
        return 0.0

    all_points = []
    valid_segmentations = []

    for seg in segmentations:
        if isinstance(seg, list) and len(seg) >= 6 and len(seg) % 2 == 0:
            points = np.array(seg, dtype=np.float32).reshape(-1, 2)
            all_points.extend(points.tolist())
            valid_segmentations.append(points)

    if not all_points:
        return 0.0

    all_points_np = np.array(all_points, dtype=np.float32)

    min_x, min_y = np.min(all_points_np, axis=0)
    max_x, max_y = np.max(all_points_np, axis=0)

    offset_x = -math.floor(min_x)
    offset_y = -math.floor(min_y)

    height = int(math.ceil(max_y) - math.floor(min_y))
    width = int(math.ceil(max_x) - math.floor(min_x))

    height = max(1, height)
    width = max(1, width)

    mask = np.zeros((height, width), dtype=np.uint8)

    for points in valid_segmentations:
        shifted_points = np.copy(points)
        shifted_points[:, 0] += offset_x
        shifted_points[:, 1] += offset_y

        # AnyLabeling은 round 후 int32로 fillPoly
        points_int = np.round(shifted_points).astype(np.int32)
        cv2.fillPoly(mask, [points_int], 1)

    total_area = np.sum(mask)
    return float(total_area)


def convert_labelme_dir_to_coco_anylabeling(input_dir, output_json):
    coco = get_coco_data_anylabeling()

    category_name_to_id = {}
    next_category_id = 1
    next_image_id = 1
    next_annotation_id = 1

    json_files = sorted(glob.glob(os.path.join(input_dir, "*.json")))

    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        image_path = find_image_for_json(input_dir, json_file, data)
        if image_path is None:
            print(f"[WARN] image not found for: {json_file}")
            continue

        # LabelMe json 내부 값 우선, 없으면 실제 이미지 읽기
        width = data.get("imageWidth")
        height = data.get("imageHeight")
        if width is None or height is None:
            width, height = get_image_size(image_path)

        image_info = {
            "id": next_image_id,
            "file_name": os.path.basename(image_path),
            "width": int(width),
            "height": int(height),
            "license": 1,
            "date_captured": "",
        }
        coco["images"].append(image_info)

        for shape in data.get("shapes", []):
            if shape.get("shape_type") != "polygon":
                continue

            label = shape.get("label")
            points = shape.get("points", [])

            if not label or len(points) < 3:
                continue

            # 라벨 처리 없이 그대로 category name 사용
            if label not in category_name_to_id:
                category_name_to_id[label] = next_category_id
                coco["categories"].append({
                    "id": next_category_id,
                    "name": label,
                    "supercategory": "",
                })
                next_category_id += 1

            category_id = category_name_to_id[label]
            segmentations = points_to_segmentation(points)  # list[list[float]]
            bbox = get_min_enclosing_bbox_anylabeling(segmentations)
            area = calculate_polygon_area_anylabeling(segmentations)

            annotation = {
                "id": next_annotation_id,
                "image_id": next_image_id,
                "category_id": category_id,
                "segmentation": segmentations,
                "area": area,
                "bbox": bbox,
                "iscrowd": 0,
            }
            coco["annotations"].append(annotation)
            next_annotation_id += 1

        next_image_id += 1

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)

    print(f"Saved: {output_json}")
    print(f"images={len(coco['images'])}, annotations={len(coco['annotations'])}, categories={len(coco['categories'])}")


if __name__ == "__main__":
    convert_labelme_dir_to_coco_anylabeling(
        "./datasets/train_remapped",
        "./datasets/annotations/instances_train.json"
    )
    convert_labelme_dir_to_coco_anylabeling(
        "./datasets/val_remapped",
        "./datasets/annotations/instances_val.json"
    )