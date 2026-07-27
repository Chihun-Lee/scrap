import os
import json
import shutil
from pathlib import Path

SPLITS = ["train", "val"]

BASE_DIR = "datasets"
ANN_DIR  = f"{BASE_DIR}/annotations"
IMG_DIR  = f"{BASE_DIR}/images"
LAB_DIR  = f"{BASE_DIR}/labels"


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def reset_label_output(split: str) -> str:
    label_root = os.path.join(LAB_DIR, split)
    cache_path = os.path.join(LAB_DIR, f"{split}.cache")

    if os.path.exists(label_root):
        shutil.rmtree(label_root)
    os.makedirs(label_root, exist_ok=True)

    if os.path.exists(cache_path):
        os.remove(cache_path)

    return label_root


# ✅ COCO → YOLO 변환
def convert_split(split: str):
    ann_path   = os.path.join(ANN_DIR, f"instances_{split}.json")
    label_root = reset_label_output(split)

    if not os.path.exists(ann_path):
        print(f"[WARN] {split}: annotation json not found -> {ann_path}")
        return

    with open(ann_path, "r") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    categories = sorted(coco["categories"], key=lambda x: x["id"])
    cat_id_to_yolo_id = {cat["id"]: i for i, cat in enumerate(categories)}

    # 기존 라벨 삭제
    n_anns = 0
    n_lines = 0
    n_skipped_poly = 0

    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0) == 1:
            continue

        seg = ann.get("segmentation")
        if not seg or not isinstance(seg, list):
            continue

        img_info = images.get(ann["image_id"])
        if img_info is None:
            continue

        w, h = img_info["width"], img_info["height"]
        file_name = img_info["file_name"]

        label_path = os.path.join(label_root, Path(file_name).with_suffix(".txt"))
        yolo_cls = cat_id_to_yolo_id[ann["category_id"]]

        with open(label_path, "a") as f:
            for poly in seg:
                if not isinstance(poly, list) or len(poly) < 6:
                    continue

                norm = []
                for i, x in enumerate(poly):
                    if i % 2 == 0:
                        norm.append(str(clamp01(float(x) / w)))
                    else:
                        norm.append(str(clamp01(float(x) / h)))

                xs = [float(v) for v in norm[0::2]]
                ys = [float(v) for v in norm[1::2]]
                if max(xs) - min(xs) <= 0 or max(ys) - min(ys) <= 0:
                    n_skipped_poly += 1
                    continue

                f.write(f"{yolo_cls} " + " ".join(norm) + "\n")
                n_lines += 1

        n_anns += 1

    print(f"[DONE] {split}: {n_anns} annotations -> {n_lines} polygons converted")
    print(f"   skipped degenerate polygons: {n_skipped_poly}")
    print(f"   labels out: {label_root}")


def create_dataset_files():
    ann_path = os.path.join(ANN_DIR, "instances_train.json")

    if not os.path.exists(ann_path):
        print(f"[WARN] annotation not found -> {ann_path}")
        return

    with open(ann_path, "r") as f:
        coco = json.load(f)

    # 🔥 category id 기준 정렬
    categories = sorted(coco["categories"], key=lambda x: x["id"])
    class_names = [cat["name"] for cat in categories]

    # classes.txt 생성
    classes_path = os.path.join(BASE_DIR, "classes.txt")
    with open(classes_path, "w", encoding="utf-8") as f:
        for name in class_names:
            f.write(name + "\n")

    print(f"[DONE] classes.txt written -> {classes_path}")

    # data.yaml 생성
    yaml_path = os.path.join(BASE_DIR, "data.yaml")

    yaml_content = f"""train: images/train
val: images/val

nc: {len(class_names)}
names: [
"""

    for name in class_names:
        yaml_content += f"    '{name}',\n"

    yaml_content += "]\n"

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"[DONE] data.yaml written -> {yaml_path}")


# ✅ 메인
def main():
    os.makedirs(LAB_DIR, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)

    for sp in SPLITS:
        convert_split(sp)    # 🔥 YOLO 변환만 수행

    create_dataset_files()


if __name__ == "__main__":
    main()
