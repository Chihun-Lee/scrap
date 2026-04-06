"""
LabelMe JSON 어노테이션 시각화 스크립트
- 원본 이미지 위에 폴리곤 마스크 + 라벨 텍스트 오버레이
- 리매핑 전 원본 라벨과 리매핑 후 병합 라벨 둘 다 표시
"""
import json
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from collections import Counter
import colorsys

# === 리매핑 테이블 (2_remap_labelme_exact.py 기준) ===
LABEL_MAP = {
    "18.Cut H-beam scrap": "beam", "21.h-beam, i-beam, and bar- shaped steel": "beam",
    "35.Base plates, I-beams, H-beams": "beam",
    "13.Sealed drum": "drum", "52.Sealed metal container": "drum", "54.Roll-type sealed container": "drum",
    "84. Unknown-Fan cover": "Fan",
    "67. Guillotine": "Guillotine",
    "74. Handler": "handler",
    "23.Mixed heavy iron scrap": "heavy iron", "60.Boiler tank": "heavy iron",
    "68. Magnet": "heavy iron", "76. Streetlight pole": "heavy iron",
    "69. LPG GAS cylinder": "LPG GAS cylinder",
    "11.Shredder": "machine", "14.Scrap automotive parts": "machine",
    "43.Textile machinery": "machine", "44.Mold machinery": "machine",
    "53.Gearbox": "machine", "61.Reducer": "machine",
    "64.Automotive Engine Parts": "machine", "65.Loom for printing": "machine",
    "66.pressed car side door": "machine", "89. Unknown-Machine": "machine",
    "24.Grating manhole cover": "mesh", "58.Rockfall protection net": "mesh",
    "63.Steel grating": "mesh", "83. Unknown-Manhole Cover": "mesh",
    "1.Laser cutting (thick plate)": "mixed steel", "10.Shredded general ferrous scrap": "mixed steel",
    "12.Worksite oxidized scrap": "mixed steel", "19.Forklift truck": "mixed steel",
    "22.Spring": "mixed steel", "25.Rebar coil scrap": "mixed steel",
    "38.Nail scrap": "mixed steel", "45.Shredded nails": "mixed steel",
    "55.Sorting Scrap Metal": "mixed steel", "56.Incinerated scrap metal": "mixed steel",
    "62.Rusty Chain": "mixed steel", "87. Unknown-Mobile stand sign": "mixed steel",
    "15.Gangform": "panel", "36.Air duct": "panel", "5.Elevator door": "panel",
    "51.Color-coated steel plate": "panel", "57.Deck reinforcement steel": "panel",
    "59.Fireproof door leaf": "panel", "6.Panels": "panel", "7.Incorner (form)": "panel",
    "77. Paint Can Lid": "panel",
    "2.Pipe_1": "pipe", "20.Galvanized steel pipe": "pipe", "26.Scaffolding pipe": "pipe",
    "33.Water supply pipe": "pipe", "37.Black steel pipe": "pipe",
    "40.Scaffolding pipe-Scaffolding platform": "pipe", "9.Housepipe": "pipe",
    "75. Plastic": "plastic",
    "28.Formwork tie pin": "rebar", "32.Rebar wire": "rebar",
    "41.Coiled reinforcing bar": "rebar", "42.Steel wire": "rebar",
    "48.Thick scrap wire": "rebar", "78. Unknown-Rebar": "rebar",
    "3.Pipe_2": "small pipe", "46.Lead pipe (copper pipe)": "small pipe",
    "31.Square steel pipe": "square pipe",
    "27.Scaffold base plate": "structure steel", "29.Structural steel shapes": "structure steel",
    "34.Clean sheet steel": "structure steel", "70. Cabinet": "structure steel",
    "71. Paint_Can": "structure steel", "79. Unknown-Panel": "structure steel",
    "80. Unknown-Square Pipe": "structure steel", "86. Unknown-Cabinet": "structure steel",
    "16.Chair": "trash", "4.Ton Bag": "trash", "72. Unknown": "trash",
    "82. Unknown-Plastic": "trash", "85. Unknown-Sorting Scrap Metal": "trash",
    "88. Unknown-Spray paint cans": "trash",
    "30.End-of-life vehicle scrap": "vehicle", "47.End-of-life vehicle shell": "vehicle",
    "8.Electronic devices": "vehicle", "81. Unknown-Vehicle Part": "vehicle",
}

# 19개 병합 클래스에 고유 색상 부여
MERGED_CLASSES = sorted(set(LABEL_MAP.values()))
def generate_colors(n):
    colors = []
    for i in range(n):
        hue = i / n
        r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        colors.append((int(r*255), int(g*255), int(b*255)))
    return colors

CLASS_COLORS = {cls: col for cls, col in zip(MERGED_CLASSES, generate_colors(len(MERGED_CLASSES)))}


def visualize_one(image_path, json_path, output_path):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    shapes = data.get("shapes", [])

    # 1) 오버레이 생성
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay, "RGBA")

    label_positions = []

    for shape in shapes:
        raw_label = shape.get("label", "")
        points = shape.get("points", [])
        if len(points) < 3:
            continue

        merged = LABEL_MAP.get(raw_label, raw_label)
        color = CLASS_COLORS.get(merged, (200, 200, 200))
        fill = color + (80,)  # RGBA with alpha

        poly = [(float(p[0]), float(p[1])) for p in points]
        draw.polygon(poly, fill=fill, outline=color)

        # centroid for label
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        cx, cy = sum(xs)/len(xs), sum(ys)/len(ys)
        label_positions.append((cx, cy, merged, color))

    # 2) 라벨 텍스트
    draw_text = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
    except:
        font = ImageFont.load_default()

    for cx, cy, text, color in label_positions:
        bbox = draw_text.textbbox((cx, cy), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx, ty = int(cx - tw/2), int(cy - th/2)
        # background
        draw_text.rectangle([tx-2, ty-2, tx+tw+2, ty+th+2], fill=(0, 0, 0, 160))
        draw_text.text((tx, ty), text, fill=(255, 255, 255), font=font)

    overlay.save(output_path, quality=85)

    # 3) 통계
    raw_labels = [s["label"] for s in shapes]
    merged_labels = [LABEL_MAP.get(l, l) for l in raw_labels]
    return Counter(merged_labels), len(shapes)


def main():
    samples = [
        ("data_check/datasets/train_data/001_13-55-11.jpg",
         "data_check/datasets/train_data/001_13-55-11.json"),
        ("data_check/datasets/train_data/013_08-12-31.jpg",
         "data_check/datasets/train_data/013_08-12-31.json"),
        ("data_check/datasets/val_data/002_13-58-56.jpg",
         "data_check/datasets/val_data/002_13-58-56.json"),
    ]

    os.makedirs("data_check/vis", exist_ok=True)

    print("=" * 60)
    print("데이터셋 요약")
    print(f"총 이미지: train 2,096장 / val 419장 = 2,515장")
    print(f"해상도: 3840 x 2160 (4K)")
    print(f"병합 클래스 수: {len(MERGED_CLASSES)}종")
    print(f"클래스: {', '.join(MERGED_CLASSES)}")
    print("=" * 60)

    for img_path, json_path in samples:
        name = os.path.basename(img_path).replace(".jpg", "")
        out_path = f"data_check/vis/{name}_labeled.jpg"

        counts, total = visualize_one(img_path, json_path, out_path)
        print(f"\n[{name}] 총 {total}개 객체")
        for cls, cnt in counts.most_common():
            print(f"  {cls}: {cnt}")
        print(f"  -> 저장: {out_path}")


if __name__ == "__main__":
    main()
