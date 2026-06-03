#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Labelme class remapping tool
- Uses exact original label strings including:
  * numeric prefix
  * dot
  * spaces
  * underscores
  * hyphens
- Remaps shapes[].label in Labelme JSON files
- remap_report.json / label_map_used.json 생성 안 함
- 이미지 복사 안 함
- 대신 output json의 imagePath를 datasets/images/{split} 기준 참조로 변경
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# =========================================================
# Exact original labels -> merged labels
# =========================================================
LABEL_MAP: Dict[str, str] = {
    "18.Cut H-beam scrap": "beam",
    "21.h-beam, i-beam, and bar- shaped steel": "beam",
    "35.Base plates, I-beams, H-beams": "beam",

    "13.Sealed drum": "drum",
    "52.Sealed metal container": "drum",
    "54.Roll-type sealed container": "drum",

    "84. Unknown-Fan cover": "Fan",

    "67. Guillotine": "Guillotine",

    "74. Handler": "handler",

    "23.Mixed heavy iron scrap": "heavy iron",
    "60.Boiler tank": "heavy iron",
    "68. Magnet": "heavy iron",
    "76. Streetlight pole": "heavy iron",

    "69. LPG GAS cylinder": "LPG GAS cylinder",

    "11.Shredder": "machine",
    "14.Scrap automotive parts": "machine",
    "43.Textile machinery": "machine",
    "44.Mold machinery": "machine",
    "53.Gearbox": "machine",
    "61.Reducer": "machine",
    "64.Automotive Engine Parts": "machine",
    "65.Loom for printing": "machine",
    "66.pressed car side door": "machine",
    "89. Unknown-Machine": "machine",

    "24.Grating manhole cover": "mesh",
    "58.Rockfall protection net": "mesh",
    "63.Steel grating": "mesh",
    "83. Unknown-Manhole Cover": "mesh",

    "1.Laser cutting (thick plate)": "mixed steel",
    "10.Shredded general ferrous scrap": "mixed steel",
    "12.Worksite oxidized scrap": "mixed steel",
    "19.Forklift truck": "mixed steel",
    "22.Spring": "mixed steel",
    "25.Rebar coil scrap": "mixed steel",
    "38.Nail scrap": "mixed steel",
    "45.Shredded nails": "mixed steel",
    "55.Sorting Scrap Metal": "mixed steel",
    "56.Incinerated scrap metal": "mixed steel",
    "62.Rusty Chain": "mixed steel",
    "87. Unknown-Mobile stand sign": "mixed steel",

    "15.Gangform": "panel",
    "36.Air duct": "panel",
    "5.Elevator door": "panel",
    "51.Color-coated steel plate": "panel",
    "57.Deck reinforcement steel": "panel",
    "59.Fireproof door leaf": "panel",
    "6.Panels": "panel",
    "7.Incorner (form)": "panel",
    "77. Paint Can Lid": "panel",

    "2.Pipe_1": "pipe",
    "20.Galvanized steel pipe": "pipe",
    "26.Scaffolding pipe": "pipe",
    "33.Water supply pipe": "pipe",
    "37.Black steel pipe": "pipe",
    "40.Scaffolding pipe-Scaffolding platform": "pipe",
    "9.Housepipe": "pipe",

    "75. Plastic": "plastic",

    "28.Formwork tie pin": "rebar",
    "32.Rebar wire": "rebar",
    "41.Coiled reinforcing bar": "rebar",
    "42.Steel wire": "rebar",
    "48.Thick scrap wire": "rebar",
    "78. Unknown-Rebar": "rebar",

    "3.Pipe_2": "small pipe",
    "46.Lead pipe (copper pipe)": "small pipe",

    "31.Square steel pipe": "square pipe",

    "27.Scaffold base plate": "structure steel",
    "29.Structural steel shapes": "structure steel",
    "34.Clean sheet steel": "structure steel",
    "70. Cabinet": "structure steel",
    "71. Paint_Can": "structure steel",
    "79. Unknown-Panel": "structure steel",
    "80. Unknown-Square Pipe": "structure steel",
    "86. Unknown-Cabinet": "structure steel",

    "16.Chair": "trash",
    "4.Ton Bag": "trash",
    "72. Unknown": "trash",
    "82. Unknown-Plastic": "trash",
    "85. Unknown-Sorting Scrap Metal": "trash",
    "88. Unknown-Spray paint cans": "trash",

    "30.End-of-life vehicle scrap": "vehicle",
    "47.End-of-life vehicle shell": "vehicle",
    "8.Electronic devices": "vehicle",
    "81. Unknown-Vehicle Part": "vehicle",
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def find_image_for_json(json_path: Path, data: dict) -> Path | None:
    image_path = data.get("imagePath")
    if image_path:
        candidate = (json_path.parent / image_path).resolve()
        if candidate.exists():
            return candidate

    stem = json_path.stem
    for ext in IMAGE_EXTS:
        candidate = json_path.with_suffix(ext)
        if candidate.exists():
            return candidate

    for ext in IMAGE_EXTS:
        candidate = json_path.parent / f"{stem}{ext}"
        if candidate.exists():
            return candidate

    return None


def remap_shapes(
    shapes: List[dict],
    label_map: Dict[str, str],
    unmapped_mode: str,
    unknown_label: str,
) -> Tuple[List[dict], Counter, Counter, Counter]:
    new_shapes: List[dict] = []
    old_counts: Counter = Counter()
    new_counts: Counter = Counter()
    unmapped_counts: Counter = Counter()

    for shape in shapes:
        old_label = shape.get("label", "")
        if not isinstance(old_label, str) or old_label == "":
            continue

        old_counts[old_label] += 1

        if old_label in label_map:
            new_shape = dict(shape)
            new_shape["label"] = label_map[old_label]
            new_shapes.append(new_shape)
            new_counts[label_map[old_label]] += 1
        else:
            unmapped_counts[old_label] += 1

            if unmapped_mode == "keep":
                new_shapes.append(shape)
                new_counts[old_label] += 1
            elif unmapped_mode == "drop":
                continue
            elif unmapped_mode == "unknown":
                new_shape = dict(shape)
                new_shape["label"] = unknown_label
                new_shapes.append(new_shape)
                new_counts[unknown_label] += 1

    return new_shapes, old_counts, new_counts, unmapped_counts


def update_image_reference(
    data: dict,
    json_path: Path,
    output_dir: Path,
    split: str,
) -> None:
    """
    output json 기준으로 ../images/{split}/파일명 을 가리키도록 imagePath를 갱신
    예:
      output_dir = datasets/train_remapped
      imagePath  = ../images/train/abc.jpg
    """
    image_path = find_image_for_json(json_path, data)
    if image_path is None:
        return

    referenced = Path("..") / "images" / split / image_path.name
    data["imagePath"] = referenced.as_posix()


def process_dataset(
    input_dir: Path,
    output_dir: Path,
    split: str,
    unmapped_mode: str,
    unknown_label: str,
    overwrite: bool,
) -> None:
    if not input_dir.exists():
        print(f"[WARN] Input directory does not exist: {input_dir}")
        return

    if output_dir.exists():
        if overwrite:
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}\n"
                f"Set overwrite = True to replace it."
            )

    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.rglob("*.json"))
    if not json_files:
        print(f"[WARN] No JSON files found in {input_dir}")
        return

    total_old_counts: Counter = Counter()
    total_new_counts: Counter = Counter()
    total_unmapped_counts: Counter = Counter()

    processed_files = 0

    for json_path in json_files:
        rel_json = json_path.relative_to(input_dir)
        out_json = output_dir / rel_json

        try:
            data = load_json(json_path)
        except Exception as e:
            print(f"[ERROR] Failed to read {json_path}: {e}")
            continue

        shapes = data.get("shapes", [])
        new_shapes, old_counts, new_counts, unmapped_counts = remap_shapes(
            shapes=shapes,
            label_map=LABEL_MAP,
            unmapped_mode=unmapped_mode,
            unknown_label=unknown_label,
        )

        data["shapes"] = new_shapes
        update_image_reference(data, json_path, output_dir, split)
        save_json(out_json, data)

        total_old_counts.update(old_counts)
        total_new_counts.update(new_counts)
        total_unmapped_counts.update(unmapped_counts)

        processed_files += 1

    print("[DONE] Exact label remapping completed.")
    print(f"Input : {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Processed JSON files: {processed_files}")

    if total_unmapped_counts:
        print("\n[INFO] Unmapped labels:")
        for label, count in total_unmapped_counts.most_common():
            print(f"  {label} -> {count}")

    print("\n[INFO] New label counts:")
    for label, count in total_new_counts.most_common():
        print(f"  {label} -> {count}")


def main() -> None:
    dataset_pairs = [
        (
            "train",
            Path("./datasets/train_data_filtered").resolve(),
            Path("./datasets/train_remapped").resolve(),
        ),
        (
            "val",
            Path("./datasets/val_data_filtered").resolve(),
            Path("./datasets/val_remapped").resolve(),
        ),
    ]

    unmapped_mode = "keep"   # "keep", "drop", "unknown"
    unknown_label = "unknown"
    overwrite = True

    for split, input_dir, output_dir in dataset_pairs:
        print("\n" + "=" * 80)
        process_dataset(
            input_dir=input_dir,
            output_dir=output_dir,
            split=split,
            unmapped_mode=unmapped_mode,
            unknown_label=unknown_label,
            overwrite=overwrite,
        )


if __name__ == "__main__":
    main()