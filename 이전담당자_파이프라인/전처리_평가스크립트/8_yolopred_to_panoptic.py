# import os
# import json
# import numpy as np
# from PIL import Image, ImageDraw
# from tqdm import tqdm

# #예측 결과인 txt 로 RGB마스크와 pred json파일 생성

# # 경로 설정
# pred_txt_dir = 'runs/segment/predict5/labels/'        # YOLO 예측 결과 폴더
# image_dir = 'dataset/images/test/'                 # 원본 이미지 폴더 (해상도 필요)
# output_mask_dir = 'PQ/panoptic_pred/'              # 예측 RGB 마스크 저장 위치
# output_json_path = 'PQ/panoptic_predictions.json'  # 예측 JSON 경로

# os.makedirs(output_mask_dir, exist_ok=True)

# # 클래스 정보 (예: coco-style)
# # id: 정수 ID, name: 클래스 이름, isthing: 1 or 0

# # 필요 시 수정


import os
import json
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

# 경로 설정 - 변경
pred_txt_dir = 'runs/segment/predict/labels/'        # YOLO 예측 결과 txt

# 경로 설정
image_dir = 'datasets/images/val/'                    # 원본 이미지
output_mask_dir = 'PQ/panoptic_pred/'                 # 저장할 RGB 마스크
output_json_path = 'PQ/panoptic_predictions.json'     # 결과 JSON

gt_instance_path = 'datasets/annotations/instances_val.json'  # GT JSON

# 디렉토리 생성
os.makedirs(output_mask_dir, exist_ok=True)

# GT instance JSON에서 categories와 image_id 매핑 불러오기
with open(gt_instance_path, 'r') as f:
    gt_data = json.load(f)

categories = gt_data['categories']
filename_to_id = {img['file_name']: img['id'] for img in gt_data['images']}

# segment_id → RGB 변환
def segment_id_to_rgb(segment_id: int):
    # id -> (R,G,B)
    r = segment_id % 256
    g = (segment_id // 256) % 256
    b = (segment_id // 65536) % 256
    return [r, g, b]

def rgb_to_segment_id(rgb):
    # (R,G,B) -> id
    r, g, b = map(int, rgb)
    return r + g * 256 + b * 256 * 256


# polygon 좌표 복원
def denormalize_poly(points, w, h):
    return [(float(points[i]) * w, float(points[i + 1]) * h) for i in range(0, len(points), 2)]

# 결과 리스트
results = []

# 예측 파일 반복
for txt_file in tqdm(sorted(os.listdir(pred_txt_dir))):
    if not txt_file.endswith('.txt'):
        continue

    image_name = os.path.splitext(txt_file)[0] + '.jpg'
    image_path = os.path.join(image_dir, image_name)

    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        continue

    image_id = filename_to_id.get(image_name, None)
    if image_id is None:
        print(f"⚠ image_id not found for {image_name}")
        continue

    with Image.open(image_path) as img:
        w, h = img.size

    mask = Image.new('RGB', (w, h), (0, 0, 0))
    draw = ImageDraw.Draw(mask)
    segments_info = []
    segment_idx = 1000

    with open(os.path.join(pred_txt_dir, txt_file), 'r') as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    # 1) 파싱 + 면적(예상치) 먼저 계산
    items = []
    for line in lines:
        parts = line.split()
        if len(parts) < 7:
            continue
        class_id = int(parts[0])
        polygon = list(map(float, parts[1:]))
        abs_polygon = denormalize_poly(polygon, w, h)

        # 면적(폴리곤 좌표로 대략 계산; temp_mask로 정확히 해도 됨)
        tmp = Image.new('L', (w, h), 0)
        ImageDraw.Draw(tmp).polygon(abs_polygon, fill=1)
        area_est = int(np.array(tmp).sum())
        if area_est <= 0:
            continue

        items.append({
            "class_id": class_id,
            "abs_polygon": [(int(x), int(y)) for (x, y) in abs_polygon],  # 정수 좌표로 고정
            "area_est": area_est
        })

    # 2) 큰 것 먼저(내림차순) → 작은 것 나중에 위로 올라오게
    items.sort(key=lambda d: d["area_est"], reverse=True)

    # 3) 그리기
    mask = Image.new('RGB', (w, h), (0, 0, 0))
    draw = ImageDraw.Draw(mask)
    segments_info = []
    segment_idx = 1000

    for it in items:
        class_id = it["class_id"]
        abs_polygon = it["abs_polygon"]

        segment_id = segment_idx
        color = segment_id_to_rgb(segment_id)        # [r,g,b]  (RGB 유지)
        draw.polygon(abs_polygon, fill=tuple(color)) # RGB로 채우기

        # bbox/area는 최종 검증 전에 임시로 기록
        temp_mask = Image.new('L', (w, h), 0)
        ImageDraw.Draw(temp_mask).polygon(abs_polygon, fill=1)
        np_mask = np.array(temp_mask)
        area = int(np.sum(np_mask))
        ys, xs = np.where(np_mask == 1)
        if xs.size == 0 or ys.size == 0:
            continue
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
        bbox = [int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)]

        segments_info.append({
            "id": segment_id,
            "category_id": class_id + 1,  # 필요에 맞게 매핑
            "bbox": bbox,
            "area": area,
            "iscrowd": 0
        })
        segment_idx += 1

    # 4) 최종 가시성 검증: PNG에 실제 색이 남아있는 segment만 유지
    mask_np = np.array(mask)  # (H,W,3) uint8
    visible_segments = []
    for seg in segments_info:
        r, g, b = segment_id_to_rgb(seg["id"])
        hit = np.any(
            (mask_np[:, :, 0] == r) &
            (mask_np[:, :, 1] == g) &
            (mask_np[:, :, 2] == b)
        )
        if hit:
            visible_segments.append(seg)
    # JSON에는 눈에 보이는 것만 기록
    segments_info = visible_segments

    # 5) 저장 경로/이름은 PNG로
    mask_name = os.path.splitext(image_name)[0] + '.png'
    out_mask_path = os.path.join(output_mask_dir, mask_name)
    mask.save(out_mask_path, format='PNG', compress_level=0)

    results.append({
        "image_id": image_id,
        "file_name": mask_name,          # ← PNG 파일명
        "segments_info": segments_info
    })


# 최종 JSON
prediction_json = {
    "annotations": results,
    "categories": categories

}

with open(output_json_path, 'w') as f:
    json.dump(prediction_json, f, indent=2)

print(f"✅ RGB 마스크 저장 완료: {output_mask_dir}")
print(f"✅ panoptic_predictions.json 저장 완료: {output_json_path}")




# categories = [
#     {"id": 1, "name": "ground", "isthing": 0},
#     {"id": 2, "name": "hbeam", "isthing": 1},
#     {"id": 3, "name": "machine", "isthing": 1},
#     {"id": 4, "name": "pipe", "isthing": 1},
#     {"id": 5, "name": "plate", "isthing": 1},
#     {"id": 6, "name": "truckbottom", "isthing": 0},
#     {"id": 7, "name": "truckwall", "isthing": 0},
# ]


# # 최종 JSON 저장
# prediction_json = {
#     "annotations": results,
#     "categories": categories
# }

# with open(output_json_path, 'w') as f:
#     json.dump(prediction_json, f, indent=2)

# print(f"✅ RGB 마스크 저장 완료: {output_mask_dir}")
# print(f"✅ panoptic_predictions.json 저장 완료: {output_json_path}")