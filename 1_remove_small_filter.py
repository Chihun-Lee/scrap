import json
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple

from PIL import Image


# =========================
# 설정
# =========================
YOLO_IMGSZ = 640
MIN_SIZE_AFTER_RESIZE = 8.0  # 리사이즈 후 bbox 최소변 기준

# input_dir, output_dir, image_ref_dir
DATASET_PAIRS = [
    (
        Path("datasets/train_data"),
        Path("datasets/train_data_filtered"),
        Path("datasets/images/train"),
    ),
    (
        Path("datasets/val_data"),
        Path("datasets/val_data_filtered"),
        Path("datasets/images/val"),
    ),
]


# =========================
# 유틸
# =========================
def get_image_size(json_data: Dict[str, Any], json_path: Path) -> Tuple[int, int]:
    """
    LabelMe JSON에서 imageWidth/imageHeight 우선 사용.
    없으면 같은 이름 이미지 또는 imagePath를 찾아서 읽음.
    """
    w = json_data.get("imageWidth")
    h = json_data.get("imageHeight")

    if isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0:
        return w, h

    candidates = []

    image_path_in_json = json_data.get("imagePath")
    if image_path_in_json:
        candidates.append((json_path.parent / image_path_in_json).resolve())

    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        candidates.append((json_path.parent / f"{json_path.stem}{ext}").resolve())

    for p in candidates:
        if p.exists():
            with Image.open(p) as im:
                return im.size

    raise FileNotFoundError(f"이미지 크기를 찾을 수 없습니다: {json_path}")


def polygon_bbox(points: List[List[float]]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def resize_scale_for_yolo(img_w: int, img_h: int, imgsz: int = 640) -> float:
    """
    Ultralytics YOLO letterbox 전 스케일 개념.
    긴 변을 imgsz에 맞춘다고 보고 scale 계산.
    """
    return min(imgsz / img_w, imgsz / img_h)


def should_keep_shape(shape: Dict[str, Any], scale: float, min_size_after_resize: float) -> bool:
    """
    polygon 객체만 검사.
    리사이즈 후 bbox 최소변이 min_size_after_resize 이상이면 유지.
    """
    shape_type = shape.get("shape_type", "polygon")
    if shape_type != "polygon":
        return True

    points = shape.get("points", [])
    if not points or len(points) < 3:
        return False

    x1, y1, x2, y2 = polygon_bbox(points)
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)

    bw_r = bw * scale
    bh_r = bh * scale

    return min(bw_r, bh_r) >= min_size_after_resize


def find_image_for_json(json_path: Path, json_data: Dict[str, Any]) -> Path | None:
    """
    JSON과 연결된 원본 이미지 경로 찾기.
    """
    image_path_in_json = json_data.get("imagePath")
    if image_path_in_json:
        p = (json_path.parent / image_path_in_json).resolve()
        if p.exists():
            return p

    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        p = (json_path.parent / f"{json_path.stem}{ext}").resolve()
        if p.exists():
            return p

    return None


def copy_images_once(input_dir: Path, image_ref_dir: Path) -> None:
    """
    train_data / val_data -> images/train / images/val 로 이미지 복사
    이미 있으면 skip
    """
    image_ref_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0

    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        for img_path in input_dir.glob(f"*{ext}"):
            dst = image_ref_dir / img_path.name
            if dst.exists():
                skipped += 1
                continue
            shutil.copy2(img_path, dst)
            copied += 1

    print(f"[IMG COPY] {input_dir.name} -> {image_ref_dir} | copied={copied}, skipped={skipped}")


def process_one_json(json_path: Path, output_dir: Path, image_ref_dir: Path) -> Tuple[int, int, int]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_w, img_h = get_image_size(data, json_path)
    scale = resize_scale_for_yolo(img_w, img_h, YOLO_IMGSZ)

    original_shapes = data.get("shapes", [])
    kept_shapes = []
    removed_shapes = []

    for shape in original_shapes:
        if should_keep_shape(shape, scale, MIN_SIZE_AFTER_RESIZE):
            kept_shapes.append(shape)
        else:
            removed_shapes.append(shape)

    data["shapes"] = kept_shapes

    # 원본 이미지 찾기
    img_path = find_image_for_json(json_path, data)
    if img_path is None or not img_path.exists():
        raise FileNotFoundError(f"연결된 이미지를 찾을 수 없습니다: {json_path}")

    # output_dir 기준 상대 imagePath로 변경
    # 예: datasets/train_data_filtered/*.json -> ../images/train/xxx.jpg
    rel_image_path = Path("..") / "images" / image_ref_dir.name / img_path.name
    data["imagePath"] = rel_image_path.as_posix()

    # 출력 폴더 생성
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON 저장
    out_json = output_dir / json_path.name
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(
        f"[OK] {json_path.name} | "
        f"orig={len(original_shapes)} kept={len(kept_shapes)} removed={len(removed_shapes)} | "
        f"img={img_w}x{img_h} scale={scale:.4f} | "
        f"imagePath={data['imagePath']}"
    )

    return len(original_shapes), len(kept_shapes), len(removed_shapes)


def process_dataset(input_dir: Path, output_dir: Path, image_ref_dir: Path) -> None:
    if not input_dir.exists():
        print(f"[WARN] 입력 폴더가 없습니다: {input_dir}")
        return

    # 이미지 먼저 복사
    copy_images_once(input_dir, image_ref_dir)

    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        print(f"[WARN] JSON 파일이 없습니다: {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"[INFO] input_dir={input_dir}")
    print(f"[INFO] output_dir={output_dir}")
    print(f"[INFO] image_ref_dir={image_ref_dir}")
    print(f"[INFO] yolo_imgsz={YOLO_IMGSZ}")
    print(f"[INFO] min_size_after_resize={MIN_SIZE_AFTER_RESIZE}px")
    print(f"[INFO] json_count={len(json_files)}")
    print("-" * 80)

    total_orig = 0
    total_kept = 0
    total_removed = 0

    for json_path in json_files:
        try:
            orig, kept, removed = process_one_json(json_path, output_dir, image_ref_dir)
            total_orig += orig
            total_kept += kept
            total_removed += removed
        except Exception as e:
            print(f"[ERROR] {json_path.name}: {e}")

    print("-" * 80)
    print(
        f"[SUMMARY] {input_dir.name} | "
        f"orig={total_orig} kept={total_kept} removed={total_removed}"
    )


def main():
    for input_dir, output_dir, image_ref_dir in DATASET_PAIRS:
        process_dataset(input_dir, output_dir, image_ref_dir)


if __name__ == "__main__":
    main()