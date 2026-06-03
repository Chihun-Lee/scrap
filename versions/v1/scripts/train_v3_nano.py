"""
Steel Scrap v3 — yolo11n-seg + overlap_mask=False (TAL 버그 회피 최종 시도)
==========================================================================
변경사항:
- 모델: yolo11s-seg → yolo11n-seg (더 작고 안정적)
- overlap_mask=False (instance overlap 처리 단순화)
- outlier 이미지 (>250 instance) 사전 제외
- scratch부터 시작 (이전 last.pt는 손상 가능성)
- batch=2 (nano 모델이라 메모리 여유)

Usage:
    conda activate scrap
    python train_v3_nano.py
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

    # Scratch부터 (작은 모델, 안정성 우선)
    model = YOLO("yolo11n-seg.pt")

    results = model.train(
        data="datasets/data.yaml",

        device="mps",
        imgsz=640,
        batch=2,

        epochs=30,
        patience=15,
        optimizer="AdamW",
        lr0=3e-4,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=2,

        freeze=10,              # backbone freeze
        amp=False,
        workers=0,
        cache=False,

        # ── augmentation 최소 ──
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
        flipud=0.0, fliplr=0.0,
        mosaic=0.0, mixup=0.0, copy_paste=0.0,
        erasing=0.0,
        auto_augment=None,

        # ── TAL 안전장치 ──
        overlap_mask=False,     # 핵심: instance overlap 단순화
        mask_ratio=4,

        max_det=300,

        project="runs/segment",
        name="v3_nano_safe",
        exist_ok=True,
        plots=True,
        save=True,
        save_period=2,
    )

    print("\n학습 완료!")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
