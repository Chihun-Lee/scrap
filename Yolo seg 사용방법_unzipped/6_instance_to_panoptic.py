import os
import json
import numpy as np
from PIL import Image
from pycocotools import mask as maskUtils

# ===== Panoptic 카테고리 (요청하신 isthing 포함) =====
CATEGORIES = [
    {"color": [123, 201, 45],  "isthing": 1, "id": 1,  "name": "mixed steel"},
    {"color": [12, 87, 233],   "isthing": 1, "id": 2,  "name": "machine"},
    {"color": [201, 34, 156],  "isthing": 1, "id": 3,  "name": "drum"},
    {"color": [78, 190, 210],  "isthing": 1, "id": 4,  "name": "panel"},
    {"color": [255, 140, 60],  "isthing": 1, "id": 5,  "name": "trash"},
    {"color": [90, 45, 200],   "isthing": 1, "id": 6,  "name": "beam"},
    {"color": [34, 220, 120],  "isthing": 1, "id": 7,  "name": "pipe"},
    {"color": [180, 60, 60],   "isthing": 1, "id": 8,  "name": "heavy iron"},
    {"color": [60, 180, 75],   "isthing": 1, "id": 9,  "name": "mesh"},
    {"color": [0, 130, 200],   "isthing": 1, "id": 10, "name": "structure steel"},
    {"color": [245, 130, 48],  "isthing": 1, "id": 11, "name": "rebar"},
    {"color": [145, 30, 180],  "isthing": 1, "id": 12, "name": "small pipe"},
    {"color": [70, 240, 240],  "isthing": 1, "id": 13, "name": "vehicle"},
    {"color": [240, 50, 230],  "isthing": 1, "id": 14, "name": "square pipe"},
    {"color": [210, 245, 60],  "isthing": 1, "id": 15, "name": "Guillotine"},
    {"color": [250, 190, 190], "isthing": 1, "id": 16, "name": "LPG GAS cylinder"},
    {"color": [0, 128, 128],   "isthing": 0, "id": 17, "name": "handler"},
    {"color": [230, 190, 255], "isthing": 1, "id": 18, "name": "plastic"},
    {"color": [170, 110, 40],  "isthing": 1, "id": 19, "name": "Fan"},
]



# ===== 경로 설정 =====
instance_json_path = 'datasets/annotations/instances_val.json'
images_dir = 'datasets/images/val/'
panoptic_json_out = 'PQ/panoptic_test.json'
panoptic_masks_dir = 'PQ/panoptic_test/'

os.makedirs(panoptic_masks_dir, exist_ok=True)

# ===== COCO instance json 읽기 =====
with open(instance_json_path, 'r') as f:
    instance_data = json.load(f)

# ⚠️ instances JSON의 categories를 쓰지 말고, 요청하신 panoptic 카테고리로 교체
categories = CATEGORIES

# ===== id <-> RGB (panoptic 규격) =====
# id -> (R,G,B)
def segment_id_to_rgb(segment_id: int):
    r = segment_id % 256
    g = (segment_id // 256) % 256
    b = (segment_id // 65536) % 256
    return [r, g, b]

# (R,G,B) -> id  (검증용 필요시)
def rgb_to_segment_id(rgb):
    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    return r + 256 * g + 256 * 256 * b

# ===== 이미지별 panoptic annotation 만들기 =====
panoptic_annotations = []
image_id_to_info = {im['id']: im for im in instance_data['images']}

# (선택) YOLO/instances category id와 panoptic id가 1:1 동일하다는 가정
# 만약 다르면, 아래 매핑을 정의해서 category_id를 변환하세요.
# yolo_to_pan = {0:1, 1:2, 2:3, ...}
# 그리고 아래에서 category_id = yolo_to_pan[ann['category_id_yolo']] 처럼 사용

for image_id, image_info in image_id_to_info.items():
    width, height = image_info['width'], image_info['height']

    # 빈 RGB 마스크 초기화
    panoptic_mask = np.zeros((height, width, 3), dtype=np.uint8)

    # segments_info 저장용 리스트
    segments_info = []

    # 이미지에 해당하는 annotation 필터링
    annos = [ann for ann in instance_data['annotations'] if ann['image_id'] == image_id]

    segment_idx = 1000  # segment id 시작 번호 (0은 배경이므로 피함)

    for ann in annos:
        category_id = ann['category_id']  # ← panoptic categories의 id와 동일하다고 가정
        segm = ann['segmentation']

        # RLE or polygon 처리
        if isinstance(segm, list):
            # polygon -> rle
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        else:
            # RLE
            rle = segm

        mask = maskUtils.decode(rle)  # (H,W) binary mask

        if mask.sum() == 0:
            continue

        # 고유 segment id
        segment_id = segment_idx

        # RGB 색상 (규격대로 RGB)
        color = segment_id_to_rgb(segment_id)

        # mask 영역에 색상 입히기 (RGB 그대로, BGR 변환 금지)
        # 벡터화로 빠르게 칠하기
        m = mask.astype(bool)
        panoptic_mask[m, 0] = color[0]  # R
        panoptic_mask[m, 1] = color[1]  # G
        panoptic_mask[m, 2] = color[2]  # B

        # bbox, area
        ys, xs = np.where(m)
        x0, y0 = xs.min(), ys.min()
        x1, y1 = xs.max(), ys.max()
        bbox = [int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)]
        area = int(mask.sum())

        segments_info.append({
            "id": segment_id,
            "category_id": category_id,
            "bbox": bbox,
            "area": area,
            "iscrowd": ann.get("iscrowd", 0)
        })

        segment_idx += 1

    # 마스크 저장 (PNG)
    mask_filename = os.path.splitext(image_info['file_name'])[0] + '.png'
    Image.fromarray(panoptic_mask).save(
        os.path.join(panoptic_masks_dir, mask_filename),
        format='PNG', compress_level=0
    )

    panoptic_annotations.append({
        "image_id": image_id,
        "file_name": mask_filename,
        "segments_info": segments_info
    })

# ===== panoptic JSON 저장 =====
panoptic_data = {
    "images": instance_data['images'],
    "annotations": panoptic_annotations,
    "categories": categories  # ← isthing 포함
}

with open(panoptic_json_out, 'w') as f:
    json.dump(panoptic_data, f, indent=2)

print(f"Panoptic JSON saved to {panoptic_json_out}")
print(f"Panoptic RGB masks saved to {panoptic_masks_dir}")
