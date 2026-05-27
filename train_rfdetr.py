"""
RF-DETR Segmentation 학습 스크립트 (맥북 MPS)
=============================================
NMS-free transformer 기반 instance segmentation.
밀집/겹침 객체에 강점이 있어 철스크랩 데이터에 적합.

Usage:
    conda activate scrap
    python train_rfdetr.py
"""

import multiprocessing


def main():
    from rfdetr import RFDETRSegNano

    # 맥북 M4 Pro (48GB) 기준 설정
    # - RFDETRSegNano: 가장 경량 모델
    # - resolution=312: pretrained weights 호환 기본값
    # - batch_size=2: MPS 메모리 고려
    # - epochs=30: 테스트 학습
    model = RFDETRSegNano()

    model.train(
        dataset_dir="datasets/rfdetr",
        epochs=30,
        batch_size=2,
        grad_accum_steps=4,       # effective batch = 2 * 4 = 8
        device="mps",
        num_workers=0,            # macOS spawn 이슈 방지
        resolution=312,           # 기본 해상도 (pretrained weights 호환)
        lr=1e-4,
        output_dir="runs/rfdetr/seg_nano_test",
    )

    print("\n학습 완료!")
    print("결과: runs/rfdetr/seg_nano_test/")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
