"""
Steel Scrap Segmentation v3 — YOLO11s-seg + Backbone Freeze (맥북 MPS)
======================================================================
Phase 1: 로컬 MPS 환경에서 빠르게 baseline 확보

전략:
- backbone freeze (layer 0-9) → 메모리 ~40% 절감, 학습 속도 ↑
- AdamW lr=3e-4 (freeze 환경에선 full train보다 LR 높게)
- imgsz=640, batch=4 (M4 Pro 48GB에서 안정)
- amp=False (MPS amp 불안정)
- workers=0 (MPS+fork 회피)

예상: ~6GB 메모리, ~15분/epoch, 30 epoch = ~7.5시간

Usage:
    conda activate scrap
    python train_v3_freeze.py
"""

import os
import multiprocessing

# MPS 환경 변수 (반드시 import torch 이전에 설정)
# - HIGH_WATERMARK_RATIO=0.0 → 제한 없음 (PyTorch 2.11에서 0.7로 설정 시 low가 1.4로 잘못 잡힘)
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def main():
    from ultralytics import YOLO
    import torch

    # MPS 가용성 확인
    assert torch.backends.mps.is_available(), "MPS not available"
    print(f"PyTorch: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")

    # 모델 로드 (yolo11s-seg, COCO 사전학습)
    model = YOLO("yolo11s-seg.pt")

    # 학습 시작
    results = model.train(
        # ── 데이터 ──────────────────────────────
        data="datasets/data.yaml",

        # ── 모델/디바이스 ─────────────────────
        device="mps",
        imgsz=640,
        batch=4,

        # ── 학습 일정 ────────────────────────
        epochs=30,
        patience=15,            # early stopping
        optimizer="AdamW",
        lr0=3e-4,               # freeze 환경 → 일반 lr보다 높게
        lrf=0.01,               # final lr = lr0 * lrf
        weight_decay=0.0005,
        warmup_epochs=2,

        # ── 메모리 절감 ───────────────────────
        freeze=10,              # backbone (layer 0-9) 전체 freeze
        amp=False,              # MPS AMP 불안정
        workers=0,              # MPS + fork 회피
        cache=False,            # 4K 이미지 캐싱 시 RAM 폭발

        # ── 데이터 증강 (소형 객체 보존 위해 보수적) ──
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        fliplr=0.5,
        mosaic=0.5,             # 200 인스턴스 환경에선 mosaic 강도 낮춤
        mixup=0.0,
        scale=0.3,

        # ── 검출 설정 (밀집 장면 대응) ─────────
        max_det=500,            # 이미지당 평균 200, 최대 1800

        # ── 출력 ──────────────────────────────
        project="runs/segment",
        name="v3_freeze_local",
        exist_ok=True,
        plots=True,
        save=True,
        save_period=5,          # 5 epoch마다 체크포인트
    )

    print("\n" + "=" * 60)
    print("학습 완료!")
    print(f"결과: runs/segment/v3_freeze_local/")
    print(f"Best weights: runs/segment/v3_freeze_local/weights/best.pt")
    print("=" * 60)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
