import json
import os

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm


CLASS_STYLE_BY_NAME = {
    "structure steel": {"color": [230, 200, 70], "isthing": 1},
    "rebar": {"color": [100, 240, 180], "isthing": 1},
    "mixed steel": {"color": [123, 45, 210], "isthing": 1},
    "panel": {"color": [80, 170, 250], "isthing": 1},
    "square pipe": {"color": [180, 60, 255], "isthing": 1},
    "trash": {"color": [190, 90, 130], "isthing": 1},
    "heavy iron": {"color": [140, 75, 200], "isthing": 1},
    "small pipe": {"color": [255, 90, 90], "isthing": 1},
    "vehicle": {"color": [70, 130, 180], "isthing": 1},
    "pipe": {"color": [255, 160, 40], "isthing": 1},
    "plastic": {"color": [120, 220, 255], "isthing": 1},
    "machine": {"color": [34, 200, 145], "isthing": 1},
    "mesh": {"color": [20, 180, 220], "isthing": 1},
    "lpg gas cylinder": {"color": [90, 255, 140], "isthing": 1},
    "handler": {"color": [200, 100, 50], "isthing": 0},
    "beam": {"color": [60, 220, 100], "isthing": 1},
    "fan": {"color": [255, 120, 200], "isthing": 1},
    "drum": {"color": [210, 120, 60], "isthing": 1},
    "guillotine": {"color": [255, 200, 120], "isthing": 1},
}

PRED_TXT_DIR = "runs/segment/predict/labels/"
IMAGE_DIR = "datasets/images/val/"
OUTPUT_MASK_DIR = "PQ/panoptic_pred/"
OUTPUT_JSON_PATH = "PQ/panoptic_predictions.json"
GT_INSTANCE_PATH = "datasets/annotations/instances_val.json"


def build_panoptic_categories(instance_categories):
    missing = [
        cat["name"] for cat in instance_categories
        if cat["name"].strip().lower() not in CLASS_STYLE_BY_NAME
    ]
    if missing:
        raise ValueError(f"No panoptic style mapping defined for categories: {missing}")

    return [
        {
            "id": cat["id"],
            "name": cat["name"],
            "color": CLASS_STYLE_BY_NAME[cat["name"].strip().lower()]["color"],
            "isthing": CLASS_STYLE_BY_NAME[cat["name"].strip().lower()]["isthing"],
        }
        for cat in instance_categories
    ]


def segment_id_to_rgb(segment_id):
    r = segment_id % 256
    g = (segment_id // 256) % 256
    b = (segment_id // 65536) % 256
    return [r, g, b]


def denormalize_poly(points, width, height):
    return [
        (float(points[i]) * width, float(points[i + 1]) * height)
        for i in range(0, len(points), 2)
    ]


os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)

with open(GT_INSTANCE_PATH, "r", encoding="utf-8") as f:
    gt_data = json.load(f)

categories = build_panoptic_categories(gt_data["categories"])
filename_to_id = {img["file_name"]: img["id"] for img in gt_data["images"]}
YOLO_CLASS_TO_CATEGORY_ID = {
    idx: cat["id"] for idx, cat in enumerate(gt_data["categories"])
}

results = []

for txt_file in tqdm(sorted(os.listdir(PRED_TXT_DIR))):
    if not txt_file.endswith(".txt"):
        continue

    image_name = os.path.splitext(txt_file)[0] + ".jpg"
    image_path = os.path.join(IMAGE_DIR, image_name)
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        continue

    image_id = filename_to_id.get(image_name)
    if image_id is None:
        print(f"image_id not found for {image_name}")
        continue

    with Image.open(image_path) as img:
        width, height = img.size

    with open(os.path.join(PRED_TXT_DIR, txt_file), "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    items = []
    for line in lines:
        parts = line.split()
        if len(parts) < 7:
            continue

        class_id = int(parts[0])
        if class_id not in YOLO_CLASS_TO_CATEGORY_ID:
            print(f"Unknown YOLO class_id {class_id} in {txt_file}")
            continue

        polygon = list(map(float, parts[1:]))
        abs_polygon = denormalize_poly(polygon, width, height)

        tmp = Image.new("L", (width, height), 0)
        ImageDraw.Draw(tmp).polygon(abs_polygon, fill=1)
        area_est = int(np.array(tmp).sum())
        if area_est <= 0:
            continue

        items.append(
            {
                "class_id": class_id,
                "abs_polygon": [(int(x), int(y)) for (x, y) in abs_polygon],
                "area_est": area_est,
            }
        )

    items.sort(key=lambda item: item["area_est"], reverse=True)

    mask = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(mask)
    segments_info = []
    segment_idx = 1000

    for item in items:
        class_id = item["class_id"]
        abs_polygon = item["abs_polygon"]
        segment_id = segment_idx
        draw.polygon(abs_polygon, fill=tuple(segment_id_to_rgb(segment_id)))

        temp_mask = Image.new("L", (width, height), 0)
        ImageDraw.Draw(temp_mask).polygon(abs_polygon, fill=1)
        np_mask = np.array(temp_mask)
        area = int(np_mask.sum())
        ys, xs = np.where(np_mask == 1)
        if xs.size == 0 or ys.size == 0:
            continue

        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
        bbox = [int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)]

        segments_info.append(
            {
                "id": segment_id,
                "category_id": YOLO_CLASS_TO_CATEGORY_ID[class_id],
                "bbox": bbox,
                "area": area,
                "iscrowd": 0,
            }
        )
        segment_idx += 1

    mask_np = np.array(mask)
    visible_segments = []
    for seg in segments_info:
        r, g, b = segment_id_to_rgb(seg["id"])
        hit = np.any(
            (mask_np[:, :, 0] == r)
            & (mask_np[:, :, 1] == g)
            & (mask_np[:, :, 2] == b)
        )
        if hit:
            visible_segments.append(seg)

    mask_name = os.path.splitext(image_name)[0] + ".png"
    out_mask_path = os.path.join(OUTPUT_MASK_DIR, mask_name)
    mask.save(out_mask_path, format="PNG", compress_level=0)

    results.append(
        {
            "image_id": image_id,
            "file_name": mask_name,
            "segments_info": visible_segments,
        }
    )

prediction_json = {
    "annotations": results,
    "categories": categories,
}

with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(prediction_json, f, indent=2)

print(f"RGB masks saved to {OUTPUT_MASK_DIR}")
print(f"Panoptic prediction JSON saved to {OUTPUT_JSON_PATH}")
