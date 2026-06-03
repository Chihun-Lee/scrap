"""
Steel Scrap v3 — Resume from epoch 5 (mosaic 비활성)
=====================================================
이전 학습이 epoch 6 진행 중 Ultralytics TAL assigner shape mismatch 버그로 종료.
원인: mosaic augmentation + 다수 인스턴스 조합에서 발생하는 알려진 버그.

해결: mosaic=0, copy_paste=0, augmentation 보수적으로 → 마지막 체크포인트에서 resume

Usage:
    conda activate scrap
    python train_v3_resume.py
"""

import os
import multiprocessing

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def main():
    from ultralytics import YOLO
    import torch

    print(f"PyTorch: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")

    # 마지막 체크포인트에서 시작 (5 epoch 완료 상태)
    checkpoint = "runs/segment/runs/segment/v3_freeze_local/weights/last.pt"
    model = YOLO(checkpoint)

    results = model.train(
        # ── 데이터 ──────────────────────────────
        data="datasets/data.yaml",

        # ── 모델/디바이스 ─────────────────────
        device="mps",
        imgsz=640,
        batch=4,

        # ── 학습 일정 ────────────────────────
        epochs=30,
        patience=15,
        optimizer="AdamW",
        lr0=3e-4,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=2,

        # ── 메모리 절감 ───────────────────────
        freeze=10,
        amp=False,
        workers=0,
        cache=False,

        # ── 데이터 증강 (TAL 버그 회피: mosaic OFF) ──
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        fliplr=0.5,
        mosaic=0.0,             # ← 핵심: TAL shape mismatch 회피
        mixup=0.0,
        copy_paste=0.0,
        scale=0.2,              # 더 보수적으로
        translate=0.05,
        degrees=0.0,
        shear=0.0,

        # ── 검출 설정 ─────────────────────────
        max_det=500,

        # ── 출력 ──────────────────────────────
        project="runs/segment",
        name="v3_freeze_local",
        exist_ok=True,
        plots=True,
        save=True,
        save_period=2,          # 더 자주 저장 (안전장치)

        # ── Resume ──
        resume=False,           # 가중치는 로드, 다만 epoch 카운터는 새로 시작
    )

    print("\n" + "=" * 60)
    print("학습 완료!")
    print(f"결과: runs/segment/runs/segment/v3_freeze_local/")
    print("=" * 60)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
