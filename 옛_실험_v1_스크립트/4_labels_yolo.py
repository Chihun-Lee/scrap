import os
import json
import shutil
from pathlib import Path

SPLITS = ["train", "val"]

BASE_DIR = "datasets"
ANN_DIR  = f"{BASE_DIR}/annotations"
IMG_DIR  = f"{BASE_DIR}/images"
LAB_DIR  = f"{BASE_DIR}/labels"


# ✅ 이미지 복사 (COCO 기준으로 필요한 것만)
def copy_images(split: str):
    ann_path = os.path.join(ANN_DIR, f"instances_{split}.json")
    src_dir = os.path.join(BASE_DIR, f"{split}_data")
    dst_dir = os.path.join(IMG_DIR, split)

    if not os.path.exists(ann_path):
        print(f"⚠️  {split}: annotation 없음 -> {ann_path}")
        return

    if not os.path.exists(src_dir):
        print(f"⚠️  {split}: source dir 없음 -> {src_dir}")
        return

    os.makedirs(dst_dir, exist_ok=True)

    with open(ann_path, "r") as f:
        coco = json.load(f)

    copied = 0
    missing = 0

    for img in coco["images"]:
        file_name = img["file_name"]

        src_path = os.path.join(src_dir, file_name)
        dst_path = os.path.join(dst_dir, file_name)

        if not os.path.exists(src_path):
            print(f"[WARN] missing image: {src_path}")
            missing += 1
            continue

        if not os.path.exists(dst_path):
            shutil.copy2(src_path, dst_path)
            copied += 1

    print(f"📦 {split}: {copied} images copied / {missing} missing -> {dst_dir}")


# ✅ COCO → YOLO 변환
def convert_split(split: str):
    ann_path   = os.path.join(ANN_DIR, f"instances_{split}.json")
    label_root = os.path.join(LAB_DIR, split)

    os.makedirs(label_root, exist_ok=True)

    if not os.path.exists(ann_path):
        print(f"⚠️  {split}: annotation json이 없습니다 -> {ann_path}")
        return

    with open(ann_path, "r") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    cat_id_to_yolo_id = {cat["id"]: i for i, cat in enumerate(coco["categories"])}

    # 기존 라벨 삭제
    for img in images.values():
        lp = os.path.join(label_root, Path(img["file_name"]).with_suffix(".txt"))
        if os.path.exists(lp):
            os.remove(lp)

    n_anns = 0
    n_lines = 0

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
                        norm.append(str(float(x) / w))
                    else:
                        norm.append(str(float(x) / h))

                f.write(f"{yolo_cls} " + " ".join(norm) + "\n")
                n_lines += 1

        n_anns += 1

    print(f"✅ {split}: {n_anns} annotations → {n_lines} polygons 변환 완료")
    print(f"   labels out: {label_root}")



def create_dataset_files():
    ann_path = os.path.join(ANN_DIR, "instances_train.json")

    if not os.path.exists(ann_path):
        print(f"⚠️ annotation 없음 → {ann_path}")
        return

    with open(ann_path, "r") as f:
        coco = json.load(f)

    # 🔥 category id 기준 정렬
    categories = sorted(coco["categories"], key=lambda x: x["id"])

    class_names = [cat["name"] for cat in categories]

    # =========================
    # classes.txt 생성
    # =========================
    classes_path = os.path.join(BASE_DIR, "classes.txt")

    with open(classes_path, "w", encoding="utf-8") as f:
        for name in class_names:
            f.write(name + "\n")

    print(f"[DONE] classes.txt 생성 → {classes_path}")

    # =========================
    # data.yaml 생성
    # =========================
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

    print(f"[DONE] data.yaml 생성 → {yaml_path}")

# ✅ 메인
def main():
    os.makedirs(LAB_DIR, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)

    for sp in SPLITS:
        copy_images(sp)      # 🔥 JSON 기준으로 이미지 복사
        convert_split(sp)    # 🔥 YOLO 변환

    #config 파일 생성
    create_dataset_files()

if __name__ == "__main__":
    main()