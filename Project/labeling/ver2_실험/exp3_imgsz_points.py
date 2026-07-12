# -*- coding: utf-8 -*-
"""3순위: 입력 이미지 크기 × Polygon Point 단순화 그리드 — 속도·성능 균형점 탐색.

그리드: imgsz {960, 1280, 1600, 1920} × RDP eps {0(원본), 4(≈1 proto cell, 원본 12px), 8(원본 24px)}px@1280.
소형 컷오프는 2순위 잠정 기준(sqrt(area) 8px@1280, 세장형 24px 예외)을 고정 적용.
측정: Box/Mask mAP(50, 50-95), 추론 속도(ms/장), 학습 peak GPU 메모리.

사용 (클러스터):
  python exp3_imgsz_points.py --epochs 60                 # 스크리닝 (단축 에폭)
  python exp3_imgsz_points.py --imgsz 1280 1600 --eps 4 --epochs 300   # 상위 조합 확정 학습
"""
import argparse
import csv
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_CSV = os.path.join(HERE, "exp3_results.csv")
BASE_CUT = 8.0
KEEP_ELONGATED = 24.0


def ensure_dataset(eps):
    name = "cut8" if eps == 0 else "cut8_rdp{:g}".format(eps)
    yaml_path = os.path.join(HERE, "datasets_exp", name, "data.yaml")
    if not os.path.exists(yaml_path):
        cmd = [sys.executable, os.path.join(HERE, "prepare_yolo_dataset.py"),
               "--out", name, "--min-sqrt-area", str(BASE_CUT),
               "--keep-elongated", str(KEEP_ELONGATED), "--rdp-eps", str(eps)]
        subprocess.run(cmd, check=True)
    return name, yaml_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgsz", type=int, nargs="+", default=[960, 1280, 1600, 1920])
    ap.add_argument("--eps", type=float, nargs="+", default=[0, 4, 8])
    ap.add_argument("--model", default="yolo26x-seg.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=-1)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO

    rows = []
    for eps in args.eps:
        name, yaml_path = ensure_dataset(eps)
        for imgsz in args.imgsz:
            run_name = "exp3_{}_sz{}_e{}".format(name, imgsz, args.epochs)
            print("=== [{}] 학습 시작 ===".format(run_name))
            model = YOLO(args.model)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            model.train(data=yaml_path, imgsz=imgsz, epochs=args.epochs, batch=args.batch,
                        optimizer="AdamW", lr0=0.001, close_mosaic=10, device=args.device,
                        project=os.path.join(HERE, "runs"), name=run_name, exist_ok=True)
            peak_gb = torch.cuda.max_memory_reserved() / 1e9 if torch.cuda.is_available() else 0.0
            m = model.val(data=yaml_path, imgsz=imgsz, device=args.device)
            row = {
                "variant": name, "rdp_eps": eps, "imgsz": imgsz, "epochs": args.epochs,
                "box_map50": round(m.box.map50, 4), "box_map5095": round(m.box.map, 4),
                "mask_map50": round(m.seg.map50, 4), "mask_map5095": round(m.seg.map, 4),
                "infer_ms": round(sum(m.speed.values()), 2), "peak_vram_gb": round(peak_gb, 1),
                "run": run_name,
            }
            exists = os.path.exists(RESULTS_CSV)
            with open(RESULTS_CSV, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                if not exists:
                    w.writeheader()
                w.writerow(row)
            rows.append(row)
            print(row)

    try:
        sys.path.insert(0, os.path.expanduser("~/Code/클러스터/cluster-notify"))
        from notify import training_complete
        best = max(rows, key=lambda r: r["mask_map50"])
        training_complete("scrap", "exp3 그리드 완료 ({}런)".format(len(rows)),
                          "best: {} mask mAP50={}".format(best["run"], best["mask_map50"]))
    except Exception:
        pass


if __name__ == "__main__":
    main()
