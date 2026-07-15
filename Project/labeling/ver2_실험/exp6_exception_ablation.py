# -*- coding: utf-8 -*-
"""세장형 예외 정책 ablation — itivai 확인요청(2026-07-14) 3번 학습 검증.

exp5(stride4 생존 시뮬레이션)에서 shoelace 두께 2px@1280(원본 6px) 미만 세장형은
GT가 절반 이상 조각나는 것을 확인했다. 컷오프 10px 고정 하에 예외 정책 3종 비교:

  cut10_noexc : 면적 기준만 (예외 없음)
  cut10       : + 긴변>=24px 예외 (현행, exp2와 공유)
  cut10_w2    : + 긴변>=24px & shoelace 두께>=2px 예외 (exp5 정책 C2)

사용 (클러스터, conda activate scrap):
  python exp6_exception_ablation.py                  # 전체
  python exp6_exception_ablation.py --variants cut10_w2 --epochs 100

평가: 자기 val + 공통 val(무필터 base) + 세장형 클래스별 mask AP50 → exp6_results.csv
"""
import argparse
import csv
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_CSV = os.path.join(HERE, "exp6_results.csv")
THIN_CLASSES = ("rebar", "pipe", "small pipe", "square pipe", "mesh")

VARIANTS = {
    "cut10_noexc": {"min-sqrt-area": 10, "keep-elongated": 0},
    "cut10": {"min-sqrt-area": 10, "keep-elongated": 24},
    "cut10_w2": {"min-sqrt-area": 10, "keep-elongated": 24, "min-elongated-width": 2},
}


def ensure_dataset(name, params):
    yaml_path = os.path.join(HERE, "datasets_exp", name, "data.yaml")
    if not os.path.exists(yaml_path):
        cmd = [sys.executable, os.path.join(HERE, "prepare_yolo_dataset.py"), "--out", name]
        for k, v in params.items():
            cmd += ["--" + k, str(v)]
        subprocess.run(cmd, check=True)
    return yaml_path


def thin_class_ap(metrics):
    """SegmentMetrics → {클래스: mask AP50} (세장형 클래스만)."""
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


def train_one(name, yaml_path, base_yaml, args):
    import torch
    from ultralytics import YOLO
    # exp2에서 이미 같은 조건으로 학습한 변형(cut10)은 재학습 없이 가중치 재평가
    reuse = os.path.join(HERE, "runs", "exp2_{}_{}_e{}".format(
        name, os.path.splitext(args.model)[0], args.epochs), "weights", "best.pt")
    if os.path.exists(reuse):
        print("[{}] exp2 가중치 재사용: {}".format(name, reuse))
        model = YOLO(reuse)
        run_name = "reuse:" + os.path.relpath(reuse, HERE)
        peak_gb = 0.0
    else:
        run_name = "exp6_{}_{}_e{}".format(name, os.path.splitext(args.model)[0], args.epochs)
        model = YOLO(args.model)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        model.train(data=yaml_path, imgsz=args.imgsz, epochs=args.epochs, batch=args.batch,
                    optimizer="AdamW", lr0=0.001, close_mosaic=10, device=args.device,
                    project=os.path.join(HERE, "runs"), name=run_name, exist_ok=True)
        peak_gb = torch.cuda.max_memory_reserved() / 1e9 if torch.cuda.is_available() else 0.0
    m = model.val(data=yaml_path, imgsz=args.imgsz, device=args.device)
    mb = model.val(data=base_yaml, imgsz=args.imgsz, device=args.device)  # 공통 무필터 val
    row = {
        "variant": name, "model": args.model, "imgsz": args.imgsz, "epochs": args.epochs,
        "box_map50": round(m.box.map50, 4), "mask_map50": round(m.seg.map50, 4),
        "mask_map5095": round(m.seg.map, 4),
        "mask_map50_commonval": round(mb.seg.map50, 4),
        "peak_vram_gb": round(peak_gb, 1), "run": run_name,
    }
    for k, v in thin_class_ap(mb).items():   # 세장형 클래스 AP는 공통 val 기준
        row[k] = v
    return row


def append_csv(row):
    exists = os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS.keys()))
    ap.add_argument("--model", default="yolo26x-seg.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=-1)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    base_yaml = os.path.join(HERE, "datasets_exp", "base", "data.yaml")
    if not os.path.exists(base_yaml):
        subprocess.run([sys.executable, os.path.join(HERE, "prepare_yolo_dataset.py"),
                        "--out", "base"], check=True)
    rows = []
    for name in args.variants:
        yaml_path = ensure_dataset(name, VARIANTS[name])
        print("=== [{}] 학습 시작 ===".format(name))
        row = train_one(name, yaml_path, base_yaml, args)
        append_csv(row)
        rows.append(row)
        print(row)

    try:
        sys.path.insert(0, os.path.expanduser("~/Code/클러스터/cluster-notify"))
        from notify import training_complete
        summary = " / ".join("{}: mask mAP50={}".format(r["variant"], r["mask_map50"]) for r in rows)
        training_complete("scrap", "exp6 세장형 예외 ablation 완료", summary)
    except Exception:
        pass


if __name__ == "__main__":
    main()
