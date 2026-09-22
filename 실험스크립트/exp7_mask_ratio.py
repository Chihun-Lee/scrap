# -*- coding: utf-8 -*-
"""exp7: mask_ratio=1 검증 — 라벨링 기준 대신 학습 설정으로 얇은 객체를 살리는 대안.

exp5에서 확인한 얇은 객체 GT 소실의 원인은 seg loss의 GT ×1/4 다운샘플(mask_ratio=4).
ultralytics 8.4.x는 GT가 proto보다 크면 proto를 GT 해상도로 업샘플해 loss를 계산하므로,
mask_ratio=1이면 두께 원본 3px까지 GT가 보존된다. cut10 데이터 고정, exp2 cut10과 비교.

  python exp7_mask_ratio.py --device 7                # 기본 (mask_ratio 1)
  python exp7_mask_ratio.py --device 7 --batch 2      # OOM 시

판독: exp2 cut10(mask_ratio=4) 대비 공통 val mask mAP50·세장형 클래스 AP 비교.
"""
import argparse
import csv
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_CSV = os.path.join(HERE, "exp7_results.csv")
THIN_CLASSES = ("rebar", "pipe", "small pipe", "square pipe", "mesh")


def thin_class_ap(metrics):
    out = {}
    try:
        names = metrics.names
        idx = metrics.seg.ap_class_index
        ap50 = metrics.seg.ap50
        for i, ci in enumerate(idx):
            cname = names[int(ci)]
            if cname in THIN_CLASSES:
                out["ap50_" + cname.replace(" ", "_")] = round(float(ap50[i]), 4)
    except Exception as e:
        out["ap50_error"] = str(e)[:60]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask-ratio", type=int, default=1)
    ap.add_argument("--variant", default="cut10")
    ap.add_argument("--model", default="yolo26x-seg.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="7")
    args = ap.parse_args()

    yaml_path = os.path.join(HERE, "datasets_exp", args.variant, "data.yaml")
    base_yaml = os.path.join(HERE, "datasets_exp", "base", "data.yaml")
    for p in (yaml_path, base_yaml):
        if not os.path.exists(p):
            sys.exit("데이터셋 없음: {} — prepare_yolo_dataset.py 먼저".format(p))

    import torch
    from ultralytics import YOLO
    run_name = "exp7_{}_mr{}_{}_e{}".format(args.variant, args.mask_ratio,
                                            os.path.splitext(args.model)[0], args.epochs)
    model = YOLO(args.model)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model.train(data=yaml_path, imgsz=args.imgsz, epochs=args.epochs, batch=args.batch,
                mask_ratio=args.mask_ratio, optimizer="AdamW", lr0=0.001, close_mosaic=10,
                device=args.device, project=os.path.join(HERE, "runs"), name=run_name,
                exist_ok=True)
    peak_gb = torch.cuda.max_memory_reserved() / 1e9 if torch.cuda.is_available() else 0.0
    m = model.val(data=yaml_path, imgsz=args.imgsz, device=args.device)
    mb = model.val(data=base_yaml, imgsz=args.imgsz, device=args.device)
    row = {
        "variant": args.variant, "mask_ratio": args.mask_ratio, "model": args.model,
        "imgsz": args.imgsz, "epochs": args.epochs,
        "box_map50": round(m.box.map50, 4), "mask_map50": round(m.seg.map50, 4),
        "mask_map5095": round(m.seg.map, 4),
        "mask_map50_commonval": round(mb.seg.map50, 4),
        "peak_vram_gb": round(peak_gb, 1), "run": run_name,
    }
    for k, v in thin_class_ap(mb).items():
        row[k] = v
    exists = os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)
    print(row)
    # 완료 마커 — 모니터링 세션이 폴링해 Claude 앱 푸시로 알림
    with open(os.path.join(HERE, "exp7_mask_ratio.done"), "w") as f:
        f.write("{}: mask mAP50={} (commonval {})\n".format(
            run_name, row["mask_map50"], row["mask_map50_commonval"]))


if __name__ == "__main__":
    main()
