import os
import json
import numpy as np
from PIL import Image
from pycocotools import mask as maskUtils
import cv2
import re
from pathlib import Path


# 경로 설정
BASE_DIR = Path(__file__).resolve().parent
instance_json_path = BASE_DIR / 'datasets' / 'annotations' / 'instances_val.json'
images_dir = BASE_DIR / 'datasets' / 'images' / 'val'
output_vis_dir = BASE_DIR / 'GT_visualization'
os.makedirs(output_vis_dir, exist_ok=True)

# 클래스별 고유 색은 유지하고, id 매핑만 instances_val.json 기준으로 맞춘다.
CLASS_NAME_TO_COLOR = {
    'handler': (200, 100, 50),
    'rebar': (100, 240, 180),
    'structure steel': (230, 200, 70),
    'mixed steel': (123, 45, 210),
    'heavy iron': (140, 75, 200),
    'panel': (80, 170, 250),
    'square pipe': (180, 60, 255),
    'mesh': (20, 180, 220),
    'trash': (190, 90, 130),
    'pipe': (255, 160, 40),
    'small pipe': (255, 90, 90),
    'vehicle': (70, 130, 180),
    'plastic': (120, 220, 255),
    'machine': (34, 200, 145),
    'drum': (210, 120, 60),
    'lpg gas cylinder': (90, 255, 140),
    'beam': (60, 220, 100),
    'fan': (255, 120, 200),
    'guillotine': (255, 200, 120),
}

# -----------------------------
# 그리기 우선순위 (작을수록 먼저=뒤 배경)
# -----------------------------
PRIORITY = {
    9: 0,   # Cargo Area: 맨 먼저 (배경)  ← 필요 시 조정
    # 13: 1,
}
DEFAULT_PRIORITY = 100

# -----------------------------
# 투명도 설정
# -----------------------------
GLOBAL_ALPHA = 0.5
ALPHA_PER_CLASS = {
    # 10: 0.25,
    11: 0.2,  # Ground
}

# 라벨 스타일
# ===== 라벨 스타일 (위쪽 설정부에 추가/수정) =====
LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_FONT_SCALE = 0.6
LABEL_THICKNESS = 2

LABEL_TEXT_COLOR = (255, 255, 255)  # 흰색 텍스트
LABEL_STROKE_COLOR = (0, 0, 0)      # 텍스트 외곽선(선택)
LABEL_STROKE_THICKNESS = 0          # 0이면 비활성, 2~3 추천

LABEL_BG_COLOR = (0, 0, 0)          # 검은 배경
LABEL_BG_ALPHA = 0.6                # 배경 투명도 (0~1)
LABEL_BG_PADDING = 4                # 텍스트 주변 여백(px)


DRAW_OUTLINE = False
OUTLINE_THICKNESS = 2


DRAW_LABELS = False

def clean_label(raw: str) -> str:
    # 앞쪽의 "숫자 + (.,:,-) + 공백" 패턴 제거 (예: "31.Square", "74: Handler", "5- Panels")
    s = re.sub(r'^\s*\d+\s*[\.\-:]\s*', '', raw)
    # 혹시 또 남아있는 선행 숫자/언더스코어 제거 (예외 케이스)
    s = re.sub(r'^\s*[\d_]+\s*', '', s)
    return s.strip()


# -----------------------------
# JSON 로드 & 카테고리명 매핑
# -----------------------------
with open(instance_json_path, 'r') as f:
    instance_data = json.load(f)

# instance_data['categories']에서 id->name 매핑 생성
CAT_ID_TO_NAME = {c['id']: c['name'] for c in instance_data.get('categories', [])}

missing_colors = [
    c['name'] for c in instance_data.get('categories', [])
    if c['name'].strip().lower() not in CLASS_NAME_TO_COLOR
]
if missing_colors:
    raise ValueError(f'No color mapping defined for categories: {missing_colors}')

CATEGORY_COLORS = {
    c['id']: CLASS_NAME_TO_COLOR[c['name'].strip().lower()]
    for c in instance_data.get('categories', [])
}

# -----------------------------
# 각 이미지마다 시각화 생성
# -----------------------------
for image_info in instance_data['images']:
    image_id = image_info['id']
    file_name = image_info['file_name']
    width, height = image_info['width'], image_info['height']

    # 이미지 로드 (RGB)
    image_path = os.path.join(images_dir, file_name)
    image = np.array(Image.open(image_path).convert('RGB'))

    # 최종 색상과 알파를 담을 레이어
    paint = np.zeros_like(image, dtype=np.uint8)                 # HxWx3
    alpha_map = np.zeros((height, width), dtype=np.float32)      # HxW

    # 해당 이미지의 annotation 가져오기 & 우선순위 정렬
    annos = [ann for ann in instance_data['annotations'] if ann['image_id'] == image_id]
    annos.sort(key=lambda ann: PRIORITY.get(ann['category_id'], DEFAULT_PRIORITY))

    # 외곽선용(옵션)
    outlines = []  # (contours, color)

    # 라벨(텍스트) 정보를 잠시 저장해뒀다가 합성 이후에 그립니다.
    labels_to_draw = []  # list of (text, (x, y))

    for ann in annos:
        cid = ann['category_id']
        color = CATEGORY_COLORS.get(cid, (0, 255, 0))
        alpha = ALPHA_PER_CLASS.get(cid, GLOBAL_ALPHA)

        segm = ann['segmentation']
        if isinstance(segm, list):  # polygon
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        else:
            rle = segm  # RLE(dict) or uncompressed RLE

        mask = maskUtils.decode(rle).astype(bool)  # HxW
        if not mask.any():
            continue

        # 색/알파 채우기 (후속 객체가 앞을 덮어씀)
        paint[mask] = color
        alpha_map[mask] = float(alpha)

        # 라벨 텍스트 계산: 우선 중심(centroid), 실패하면 바운딩박스 좌상단
        m = cv2.moments(mask.astype(np.uint8))
        if m['m00'] != 0:
            cx = int(m['m10'] / m['m00'])
            cy = int(m['m01'] / m['m00'])
            text_x, text_y = cx, cy
        else:
            ys, xs = np.where(mask)
            x0, y0 = int(xs.min()), int(ys.min())
            text_x, text_y = x0, y0

        # 이미지 경계 밖으로 나가지 않도록 약간 보정
        name = CAT_ID_TO_NAME.get(cid, f'id:{cid}')
        text = clean_label(name)  # ← 이걸 사용
        
        (tw, th), baseline = cv2.getTextSize(text, LABEL_FONT, LABEL_FONT_SCALE, LABEL_THICKNESS)
        tx = np.clip(text_x - tw // 2, 0, max(0, width - tw))
        ty = np.clip(text_y + th // 2, th + baseline, height - baseline)

        labels_to_draw.append((text, (int(tx), int(ty))))

        # 외곽선 수집(옵션)
        if DRAW_OUTLINE:
            m8 = (mask.astype(np.uint8) * 255)
            contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            outlines.append((contours, color))

    # 한 번만 합성 — 누적 틴트 방지
    overlay = (image.astype(np.float32) * (1.0 - alpha_map[..., None]) +
               paint.astype(np.float32) * (alpha_map[..., None]))
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # 외곽선 그리기(옵션)
    if DRAW_OUTLINE:
        for contours, color in outlines:
            cv2.drawContours(overlay, contours, -1, color, thickness=OUTLINE_THICKNESS)

    # ★ 라벨(명칭) 그리기: 박스 없이 텍스트만, 외곽선+본문 2패스
    # ===== 라벨(명칭) 그리기: 박스 없이 탐지박스 X, 텍스트 뒤에만 반투명 사각형 =====
    if DRAW_LABELS:
        for text, (tx, ty) in labels_to_draw:
            # 텍스트 크기 측정
            (tw, th), baseline = cv2.getTextSize(text, LABEL_FONT, LABEL_FONT_SCALE, LABEL_THICKNESS)
            x0 = max(0, tx - LABEL_BG_PADDING)
            y0 = max(0, ty - th - LABEL_BG_PADDING)
            x1 = min(overlay.shape[1], tx + tw + LABEL_BG_PADDING)
            y1 = min(overlay.shape[0], ty + baseline + LABEL_BG_PADDING)

            # 배경 레이어 만들고 반투명 블렌딩
            # - ROI만 추출해서 수치연산(더 빠르고 정확)
            roi = overlay[y0:y1, x0:x1].astype(np.float32)
            if roi.size > 0:
                bg = np.full_like(roi, LABEL_BG_COLOR, dtype=np.float32)
                blended = bg * LABEL_BG_ALPHA + roi * (1.0 - LABEL_BG_ALPHA)
                overlay[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)

            # 텍스트(외곽선 → 본문 순서)
            if LABEL_STROKE_THICKNESS > 0:
                cv2.putText(overlay, text, (tx, ty),
                            LABEL_FONT, LABEL_FONT_SCALE, LABEL_STROKE_COLOR,
                            LABEL_STROKE_THICKNESS, cv2.LINE_AA)
            cv2.putText(overlay, text, (tx, ty),
                        LABEL_FONT, LABEL_FONT_SCALE, LABEL_TEXT_COLOR,
                        LABEL_THICKNESS, cv2.LINE_AA)


    # 저장
    out_path = os.path.join(output_vis_dir, file_name)
    Image.fromarray(overlay).save(out_path)

print(f"완료! 시각화된 이미지가 다음 폴더에 저장되었습니다: {output_vis_dir}")
