import json
import os

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


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

INSTANCE_JSON_PATH = "datasets/annotations/instances_val.json"
PANOPTIC_JSON_OUT = "PQ/panoptic_test.json"
PANOPTIC_MASKS_DIR = "PQ/panoptic_test/"


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


os.makedirs(PANOPTIC_MASKS_DIR, exist_ok=True)

with open(INSTANCE_JSON_PATH, "r", encoding="utf-8") as f:
    instance_data = json.load(f)

categories = build_panoptic_categories(instance_data["categories"])
panoptic_annotations = []
image_id_to_info = {image["id"]: image for image in instance_data["images"]}

for image_id, image_info in image_id_to_info.items():
    width = image_info["width"]
    height = image_info["height"]
    panoptic_mask = np.zeros((height, width, 3), dtype=np.uint8)
    segments_info = []
    annos = [ann for ann in instance_data["annotations"] if ann["image_id"] == image_id]
    segment_idx = 1000

    for ann in annos:
        category_id = ann["category_id"]
        segm = ann["segmentation"]

        if isinstance(segm, list):
            rles = mask_utils.frPyObjects(segm, height, width)
            rle = mask_utils.merge(rles)
        else:
            rle = segm

        mask = mask_utils.decode(rle).astype(bool)
        if not mask.any():
            continue

        segment_id = segment_idx
        panoptic_mask[mask] = segment_id_to_rgb(segment_id)

        ys, xs = np.where(mask)
        x0, y0 = xs.min(), ys.min()
        x1, y1 = xs.max(), ys.max()
        bbox = [int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)]
        area = int(mask.sum())

        segments_info.append(
            {
                "id": segment_id,
                "category_id": category_id,
                "bbox": bbox,
                "area": area,
                "iscrowd": ann.get("iscrowd", 0),
            }
        )
        segment_idx += 1

    mask_filename = os.path.splitext(image_info["file_name"])[0] + ".png"
    Image.fromarray(panoptic_mask).save(
        os.path.join(PANOPTIC_MASKS_DIR, mask_filename),
        format="PNG",
        compress_level=0,
    )

    panoptic_annotations.append(
        {
            "image_id": image_id,
            "file_name": mask_filename,
            "segments_info": segments_info,
        }
    )

panoptic_data = {
    "images": instance_data["images"],
    "annotations": panoptic_annotations,
    "categories": categories,
}

with open(PANOPTIC_JSON_OUT, "w", encoding="utf-8") as f:
    json.dump(panoptic_data, f, indent=2)

print(f"Panoptic JSON saved to {PANOPTIC_JSON_OUT}")
print(f"Panoptic RGB masks saved to {PANOPTIC_MASKS_DIR}")
