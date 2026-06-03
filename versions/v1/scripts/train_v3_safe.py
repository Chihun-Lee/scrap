"""
Steel Scrap v3 — Safe mode (TAL 버그 회피)
==========================================
Ultralytics TAL assigner shape mismatch 버그가 mosaic 외에도 affine augmentation에서
polygon이 깨질 때 (RuntimeWarning: invalid value encountered in matmul) 발생.

해결: 모든 augmentation 비활성, batch=1로 한 번에 처리하는 인스턴스 수 최소화.

Usage:
    conda activate scrap
    python train_v3_safe.py
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

    # 5 epoch까지 학습된 last.pt에서 가중치 로드
    checkpoint = "runs/segment/runs/segment/v3_freeze_local/weights/last.pt"
    model = YOLO(checkpoint)

    results = model.train(
        data="datasets/data.yaml",

        device="mps",
        imgsz=640,
        batch=1,                # ← 핵심 1: 인스턴스 수 최소화
        nbs=64,                 # nominal batch size, gradient accumulate

        epochs=30,
        patience=15,
        optimizer="AdamW",
        lr0=3e-4,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=2,

        freeze=10,
        amp=False,
        workers=0,
        cache=False,

        # ── 핵심 2: augmentation 모두 비활성 ──
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
        flipud=0.0, fliplr=0.0,
        mosaic=0.0, mixup=0.0, copy_paste=0.0,
        erasing=0.0,
        auto_augment=None,

        max_det=500,

        project="runs/segment",
        name="v3_freeze_local",
        exist_ok=True,
        plots=True,
        save=True,
        save_period=2,
    )

    print("\n학습 완료!")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
