# yolo_color_overlay.py
# SAM 없이 YOLO Segmentation 결과만으로 thing/stuff 시각화

import os, glob, re, json
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
INSTANCE_JSON_PATH = BASE_DIR / "datasets" / "annotations" / "instances_val.json"

CLASS_STYLE_BY_NAME = {
    "handler": {"color": [200, 100, 50], "isthing": 0},
    "rebar": {"color": [100, 240, 180], "isthing": 1},
    "structure steel": {"color": [230, 200, 70], "isthing": 1},
    "mixed steel": {"color": [123, 45, 210], "isthing": 1},
    "heavy iron": {"color": [140, 75, 200], "isthing": 1},
    "panel": {"color": [80, 170, 250], "isthing": 1},
    "square pipe": {"color": [180, 60, 255], "isthing": 1},
    "mesh": {"color": [20, 180, 220], "isthing": 1},
    "trash": {"color": [190, 90, 130], "isthing": 1},
    "pipe": {"color": [255, 160, 40], "isthing": 1},
    "small pipe": {"color": [255, 90, 90], "isthing": 1},
    "vehicle": {"color": [70, 130, 180], "isthing": 1},
    "plastic": {"color": [120, 220, 255], "isthing": 1},
    "machine": {"color": [34, 200, 145], "isthing": 1},
    "drum": {"color": [210, 120, 60], "isthing": 1},
    "lpg gas cylinder": {"color": [90, 255, 140], "isthing": 1},
    "beam": {"color": [60, 220, 100], "isthing": 1},
    "fan": {"color": [255, 120, 200], "isthing": 1},
    "guillotine": {"color": [255, 200, 120], "isthing": 1},
}

with open(INSTANCE_JSON_PATH, "r", encoding="utf-8") as f:
    instance_data = json.load(f)

missing_styles = [
    c["name"] for c in instance_data.get("categories", [])
    if c["name"].strip().lower() not in CLASS_STYLE_BY_NAME
]
if missing_styles:
    raise ValueError(f"No style mapping defined for categories: {missing_styles}")

CATEGORIES = [
    {
        "id": c["id"],
        "name": c["name"],
        "color": CLASS_STYLE_BY_NAME[c["name"].strip().lower()]["color"],
        "isthing": CLASS_STYLE_BY_NAME[c["name"].strip().lower()]["isthing"],
    }
    for c in instance_data.get("categories", [])
]


# ====== Config ======
YOLO_WEIGHTS = "runs/segment/train/weights/best.pt"
IMG_DIR   = "datasets/images/val"
OUT_DIR   = "output"
CONF, IOU = 0.05, 0.5
MIN_AREA  = 64

os.makedirs(OUT_DIR, exist_ok=True)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# ====== Helpers ======
def grow_ground_from_seed(seed_mask: np.ndarray, bgr: np.ndarray,
                          color_thresh: float = 35.0):
    """
    seed_mask: YOLO가 찍어준 Ground 마스크(업샘플 + 정리 후).
    bgr      : 원본 이미지 BGR.
    color_thresh: Lab 색공간에서 거리 임계값 (값이 크면 더 많이 퍼짐).
    """
    if seed_mask.sum() == 0:
        return seed_mask

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    seed_colors = lab[seed_mask]

    mean = seed_colors.mean(axis=0)
    diff = lab.astype(np.int16) - mean.astype(np.int16)
    dist2 = (diff ** 2).sum(axis=2)

    color_mask = dist2 < (color_thresh ** 2)

    # Ground 특성에 맞게 살짝 정리
    grown = morph_cleanup_ground(color_mask | seed_mask)
    return grown


def boxes_to_mask(boxes, H, W):
    """Segmentation mask가 없을 때 box로부터 마스크 생성."""
    m = np.zeros((H, W), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes.astype(int):
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(W-1, x2); y2 = min(H-1, y2)
        if x2 > x1 and y2 > y1:
            m[y1:y2, x1:x2] = 1
    return m.astype(bool)

def clean_name(n: str) -> str:
    return re.sub(r'^\s*\d+\s*[\.\-:_]?\s*', '', str(n)).strip()

def extract_code(raw_name: str):
    m = re.match(r'^\s*(\d+)', str(raw_name))
    return int(m.group(1)) if m else None

NAME_TO_CAT = {clean_name(c["name"]).lower(): c for c in CATEGORIES}
ID_TO_CAT   = {c["id"]: c for c in CATEGORIES if "id" in c}
CODE_TO_CAT = {c.get("code"): c for c in CATEGORIES if "code" in c}

def cat_color_bgr(cat: dict):
    r, g, b = cat["color"]
    return (b, g, r)

def draw_label_box(img_bgr, xy, text, base_size=18, center_mode=True, bg_alpha=0.70):
    """
    xy: 텍스트 기준점. center_mode=True 이면 중심에 텍스트 박스를 중앙 배치.
    """
    h, w = img_bgr.shape[:2]
    x, y = int(xy[0]), int(xy[1])

    scale = max(0.8, base_size / 24.0)
    thick = 2 if base_size >= 20 else 1

    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    pad = 4
    bw, bh = tw + pad*2, th + pad*2

    if center_mode:
        x1 = x - bw // 2
        y1 = y - bh // 2
        x2 = x1 + bw
        y2 = y1 + bh
    else:
        x1, y1 = x, y
        x2, y2 = x + bw, y - bh
        if y2 < 0:
            y1, y2 = y, y + bh

    if x1 < 0: x2 -= x1; x1 = 0
    if y1 < 0: y2 -= y1; y1 = 0
    if x2 > w:
        shift = x2 - w; x1 = max(0, x1 - shift); x2 = w
    if y2 > h:
        shift = y2 - h; y1 = max(0, y1 - shift); y2 = h

    x_left, x_right = max(0, x1), min(w, x2)
    y_top,  y_bot   = max(0, y1), min(h, y2)
    if x_right > x_left and y_bot > y_top:
        roi = img_bgr[y_top:y_bot, x_left:x_right].astype(np.float32)
        bg  = np.zeros_like(roi)
        blended = bg * bg_alpha + roi * (1.0 - bg_alpha)
        img_bgr[y_top:y_bot, x_left:x_right] = np.clip(blended, 0, 255).astype(np.uint8)

    tx = x_left + pad
    ty = y_top  + pad + th
    cv2.putText(img_bgr, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), thick, cv2.LINE_AA)
    return img_bgr

def mask_centroid(m_bool: np.ndarray):
    """이진 마스크의 중심(정수 픽셀) 반환. 비어있으면 None."""
    if m_bool is None or m_bool.sum() == 0:
        return None
    m = cv2.moments(m_bool.astype(np.uint8))
    if m["m00"] == 0:
        ys, xs = np.where(m_bool)
        return (int(xs.mean()), int(ys.mean())) if xs.size else None
    cx = int(m["m10"] / m["m00"])
    cy = int(m["m01"] / m["m00"])
    return (cx, cy)

def resolve_category(raw_name: str, class_idx: int):
    code = extract_code(raw_name)
    if code is not None and code in CODE_TO_CAT:
        return CODE_TO_CAT[code]
    cname = clean_name(raw_name).lower()
    if cname in NAME_TO_CAT:
        return NAME_TO_CAT[cname]
    return None

def upsample_mask(m: np.ndarray, out_hw):
    """
    YOLO seg mask를 원본 크기로 업샘플.
    linear interpolation + threshold 로 ground 끊김 방지.
    """
    H, W = out_hw
    m_f = cv2.resize(
        m.astype(np.float32), (W, H),
        interpolation=cv2.INTER_LINEAR
    )
    return m_f > 0.45

def morph_cleanup(m_bool):
    """일반 객체용 형태학 정리."""
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    u8 = m_bool.astype(np.uint8)
    u8 = cv2.morphologyEx(u8, cv2.MORPH_OPEN,  k3, iterations=1)
    u8 = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, k5, iterations=1)
    return (u8 > 0)

def morph_cleanup_ground(m_bool):
    """Ground(넓은 영역)용 형태학 정리: 틈 메우고 노이즈 제거."""
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    u8 = m_bool.astype(np.uint8)
    u8 = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, k7, iterations=2)  # 큰 틈 메우기
    u8 = cv2.morphologyEx(u8, cv2.MORPH_OPEN,  k5, iterations=1)  # 작은 노이즈 제거
    return (u8 > 0)

IS_THING = {c["id"]: bool(c.get("isthing", 1)) for c in CATEGORIES}

# ====== Model ======
det = YOLO(YOLO_WEIGHTS)

# ====== Inference ======
ground_cat = next(
    (c for c in CATEGORIES if clean_name(c["name"]).lower() == "ground"),
    None
)

printed_map = False
for ip in sorted(glob.glob(os.path.join(IMG_DIR, "*.*"))):
    bgr = cv2.imread(ip)
    if bgr is None:
        print(f"[skip] cannot read: {ip}")
        continue
    rgb = bgr[:, :, ::-1]
    h, w = rgb.shape[:2]

    r = det.predict(
        ip, conf=CONF, iou=IOU, verbose=False,
        agnostic_nms=True, imgsz=1280, device=device
    )[0]

    if (r.boxes is None or len(r.boxes) == 0):
        print(f"[skip] no boxes: {os.path.basename(ip)}")
        continue

    boxes = r.boxes.xyxy.cpu().numpy()
    clses = r.boxes.cls.cpu().numpy().astype(int)
    names = getattr(r, "names", None) or getattr(det.model, "names", None)

    if not printed_map:
        print("== YOLO names (id -> raw -> cleaned -> code) ==")
        if isinstance(names, dict):
            for k, v in names.items():
                print(f"id {k}: {v} -> {clean_name(v)} -> {extract_code(v)}")
        elif isinstance(names, (list, tuple)):
            for k, v in enumerate(names):
                print(f"id {k}: {v} -> {clean_name(v)} -> {extract_code(v)}")
        printed_map = True

    canvas = np.zeros_like(bgr)
    labels_to_draw = []

    # YOLO masks
    yolo_masks = None
    if getattr(r, "masks", None) is not None and r.masks is not None and r.masks.data is not None:
        yolo_masks = r.masks.data.cpu().numpy()  # [N, Hm, Wm]

    # === 1단계: thing 마스크 + ground seed 만들기 ===
    thing_union   = np.zeros((h, w), bool)
    ground_union  = np.zeros((h, w), bool)

    for i, c_idx in enumerate(clses):
        raw_name = (names.get(int(c_idx)) if isinstance(names, dict)
                    else (names[int(c_idx)] if isinstance(names, (list, tuple))
                          and 0 <= int(c_idx) < len(names)
                          else str(int(c_idx))))
        cat = resolve_category(raw_name, int(c_idx))
        is_thing = True if (cat is None) else bool(cat.get("isthing", 1))

        # 마스크 or 박스
        if yolo_masks is not None:
            m = upsample_mask(yolo_masks[i], (h, w))
        else:
            m = boxes_to_mask(np.array([boxes[i]]), h, w)

        if m.sum() < MIN_AREA:
            continue

        # === thing 처리 ===
        if is_thing:
            m_obj = morph_cleanup(m)
            if m_obj.sum() < MIN_AREA:
                continue

            # 색 & 라벨 텍스트
            if cat is not None:
                color_bgr = cat_color_bgr(cat)
                label_txt = clean_name(cat["name"])
            else:
                rng = np.random.default_rng(int(c_idx))
                color_bgr = tuple(int(x) for x in rng.integers(60, 231, size=3))
                label_txt = clean_name(raw_name)

            canvas[m_obj] = color_bgr
            thing_union |= m_obj

            ctr = mask_centroid(m_obj)
            if ctr is None:
                x1, y1, x2, y2 = boxes[i].astype(int)
                ctr = ((x1 + x2) // 2, (y1 + y2) // 2)

            font_px = int(np.clip(np.sqrt(m_obj.sum()) * 0.06, 14, 28))
            labels_to_draw.append((ctr, label_txt, font_px, True))

        # === stuff 중 Ground만 모아서 union ===
        else:
            if cat is not None and clean_name(cat["name"]).lower() == "ground":
                m_g = morph_cleanup_ground(m)
                if m_g.sum() >= MIN_AREA:
                    ground_union |= m_g

    # === 2단계: Ground union을 살짝만 확장해서 칠하기 ===
    if ground_cat is not None and ground_union.sum() > 0:
        # 살짝만 dilate 해서 끊긴 곳 메우기 (너무 크면 숫자 줄여)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        ground_mask = cv2.dilate(ground_union.astype(np.uint8), k, iterations=1) > 0

        ground_color = cat_color_bgr(ground_cat)
        canvas[ground_mask] = ground_color

    # === 3단계: 블렌딩 & 라벨 렌더링 ===
    MASK_STRENGTH = 0.7  # 색 농도 (0.7~1.0 사이에서 취향대로)
    A = bgr.astype(np.float32) / 255.0
    B = (canvas.astype(np.float32) / 255.0) * MASK_STRENGTH
    vis = 1.0 - (1.0 - A) * (1.0 - B)
    vis = (vis * 255).clip(0, 255).astype(np.uint8)

    # for (pt, txt, fpx, center_mode) in labels_to_draw:
    #     vis = draw_label_box(
    #         vis, pt, txt,
    #         base_size=fpx,
    #         center_mode=center_mode,
    #         bg_alpha=0.55
    #     )

    stem = os.path.splitext(os.path.basename(ip))[0]
    out_path = os.path.join(OUT_DIR, f"{stem}_maskcolor.jpg")
    cv2.imwrite(out_path, vis)
    print(f"[done] {stem} -> {out_path}")
