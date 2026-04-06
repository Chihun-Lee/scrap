import subprocess


SCRIPT_0 = "0_remove_cargo.py"
SCRIPT_1 = "1_remove_small_filter.py"
SCRIPT_2 = "2_remap_labelme_exact.py"
SCRIPT_3 = "3_annotations_to_instances.py"
SCRIPT_4 = "4_labels_yolo.py"


def run_script(script):
    print("\n" + "=" * 80)
    print(f"[RUN] {script}")

    result = subprocess.run(["python", script])

    if result.returncode != 0:
        raise RuntimeError(f"[ERROR] {script} failed")

    print(f"[DONE] {script}")


def main():
    run_script(SCRIPT_0)
    run_script(SCRIPT_1)
    run_script(SCRIPT_2)
    run_script(SCRIPT_3)
    run_script(SCRIPT_4)

    print("\n🔥 ALL STEPS COMPLETED")


if __name__ == "__main__":
    main()