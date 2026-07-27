import argparse
from pathlib import Path
import subprocess
import sys


# 이미 crop된 데이터를 datasets/train_cropped, datasets/val_cropped에 준비한 경우 False.
# Cargo Area 기준 crop부터 새로 수행하려면 True 또는 --run-build-cargo 를 사용한다.
RUN_BUILD_CARGO = False

SCRIPT_STEPS = [
    ("build_cargo", "0_build_cargo_dataset.py"),
    ("remove_cargo", "0_remove_cargo.py"),
    ("small_filter", "1_remove_small_filter.py"),
    ("remap_labels", "2_remap_labelme_exact.py"),
    ("instances", "3_annotations_to_instances.py"),
    ("yolo_labels", "4_labels_yolo.py"),
]

CROPPED_INPUT_DIRS = [
    Path("datasets/train_cropped"),
    Path("datasets/val_cropped"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run dataset preprocessing from cargo crop or from already-cropped data."
    )
    parser.set_defaults(run_build_cargo=RUN_BUILD_CARGO)
    parser.add_argument(
        "--run-build-cargo",
        dest="run_build_cargo",
        action="store_true",
        help="0_build_cargo_dataset.py부터 실행합니다.",
    )
    parser.add_argument(
        "--skip-build-cargo",
        dest="run_build_cargo",
        action="store_false",
        help="0_build_cargo_dataset.py를 건너뛰고 0_remove_cargo.py부터 실행합니다.",
    )
    return parser.parse_args()


def validate_cropped_inputs():
    missing = []
    for path in CROPPED_INPUT_DIRS:
        if not path.exists() or not any(path.rglob("*.json")):
            missing.append(str(path))

    if missing:
        raise FileNotFoundError(
            "RUN_BUILD_CARGO=False 이므로 crop 완료 데이터가 필요합니다. "
            f"JSON을 찾을 수 없는 폴더: {', '.join(missing)}"
        )


def run_script(script):
    print("\n" + "=" * 80)
    print(f"[RUN] {script}")

    result = subprocess.run([sys.executable, script])

    if result.returncode != 0:
        raise RuntimeError(f"[ERROR] {script} failed")

    print(f"[DONE] {script}")


def main():
    args = parse_args()

    if args.run_build_cargo:
        steps = SCRIPT_STEPS
        print("[MODE] Start from 0_build_cargo_dataset.py")
    else:
        validate_cropped_inputs()
        steps = SCRIPT_STEPS[1:]
        print("[MODE] Skip cargo crop; start from 0_remove_cargo.py")

    for _, script in steps:
        run_script(script)

    print("\n🔥 ALL STEPS COMPLETED")


if __name__ == "__main__":
    main()
