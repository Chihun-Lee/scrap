import subprocess


# =========================
# 설정
# =========================
SCRIPT_5 = "5_GT_mask.py"
SCRIPT_6 = "6_instance_to_panoptic.py"
SCRIPT_7 = "7_pred_mask.py"
SCRIPT_8 = "8_yolopred_to_panoptic.py"
SCRIPT_9 = "9_pq_calculator.py"


# =========================
# 실행 함수
# =========================
def run_simple(script):
    print(f"\n[RUN] {script}")
    result = subprocess.run(["python", script])

    if result.returncode != 0:
        raise RuntimeError(f"[ERROR] {script} failed")


# =========================
# PQ 평가 파이프라인
# =========================
def run_pq_evaluation():
    print("=" * 80)
    print("[PQ EVALUATION PIPELINE START]")

    # GT mask 생성
    run_simple(SCRIPT_5)

    # GT instance -> panoptic
    run_simple(SCRIPT_6)

    # 예측 mask 생성
    run_simple(SCRIPT_7)

    # 예측 -> panoptic
    run_simple(SCRIPT_8)

    # PQ 계산
    run_simple(SCRIPT_9)

    print("\n[DONE] PQ evaluation complete")
    print("=" * 80)


if __name__ == "__main__":
    run_pq_evaluation()