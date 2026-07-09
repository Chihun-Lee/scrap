# -*- coding: utf-8 -*-
"""2순위 재검증: 소형 인스턴스 컷오프 스윕 — YOLO26x-seg @ imgsz 1280.

원본 라벨 기준 컷오프 {무필터, 8, 10, 12, 16}px(sqrt(area)@1280 = 원본 24~48px, 세장형 예외 24px 유지)로
데이터셋 변형을 만들고 각각 학습 → val 성능/속도/메모리를 CSV로 수집한다.

사용 (클러스터, conda activate scrap):
  python exp2_train_sweep.py                         # 전체 스윕
  python exp2_train_sweep.py --cuts 8 --epochs 100   # 단일 조건
  python exp2_train_sweep.py --model yolo11s-seg.pt --epochs 30   # 스모크 테스트

주의: yolo26 + imgsz 1280 관련 업스트림 이슈가 있었으므로 ultralytics 최신(>=8.4.90) 권장.
"""
import argparse
import csv
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_CSV = os.path.join(HERE, "exp2_results.csv")


def ensure_dataset(cut, keep_elongated):
    name = "base" if cut == 0 else "cut{:g}".format(cut)
    yaml_path = os.path.join(HERE, "datasets_exp", name, "data.yaml")
    if not os.path.exists(yaml_path):
        cmd = [sys.executable, os.path.join(HERE, "prepare_yolo_dataset.py"),
               "--out", name, "--min-sqrt-area", str(cut), "--keep-elongated", str(keep_elongated)]
        subprocess.run(cmd, check=True)
    return name, yaml_path


def train_one(name, yaml_path, base_yaml, args):
    import torch
    from ultralytics import YOLO
    run_name = "exp2_{}_{}_e{}".format(name, os.path.splitext(args.model)[0], args.epochs)
    model = YOLO(args.model)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model.train(data=yaml_path, imgsz=args.imgsz, epochs=args.epochs, batch=args.batch,
                optimizer="AdamW", lr0=0.001, close_mosaic=10, device=args.device,
                project=os.path.join(HERE, "runs"), name=run_name, exist_ok=True)
    peak_gb = torch.cuda.max_memory_reserved() / 1e9 if torch.cuda.is_available() else 0.0
    m = model.val(data=yaml_path, imgsz=args.imgsz, device=args.device)
    # 공통 val(무필터 base 라벨) 재평가 — 필터별로 val 라벨이 달라지는 비교 왜곡 제거
    mb = model.val(data=base_yaml, imgsz=args.imgsz, device=args.device)
    row = {
        "variant": name, "model": args.model, "imgsz": args.imgsz, "epochs": args.epochs,
        "box_map50": round(m.box.map50, 4), "box_map5095": round(m.box.map, 4),
        "mask_map50": round(m.seg.map50, 4), "mask_map5095": round(m.seg.map, 4),
        "mask_map50_commonval": round(mb.seg.map50, 4), "box_map50_commonval": round(mb.box.map50, 4),
        "infer_ms": round(sum(m.speed.values()), 2), "peak_vram_gb": round(peak_gb, 1),
        "run": run_name,
    }
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
    ap.add_argument("--cuts", type=float, nargs="+", default=[0, 8, 10, 12, 16])
    ap.add_argument("--keep-elongated", type=float, default=24.0)
    ap.add_argument("--model", default="yolo26x-seg.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=-1, help="-1=AutoBatch")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    _, base_yaml = ensure_dataset(0, args.keep_elongated)  # 공통 val 평가용 무필터 base
    rows = []
    for cut in args.cuts:
        name, yaml_path = ensure_dataset(cut, args.keep_elongated)
        print("=== [{}] 학습 시작 ===".format(name))
        row = train_one(name, yaml_path, base_yaml, args)
        append_csv(row)
        rows.append(row)
        print(row)

    # cluster-notify (있으면 사용)
    try:
        sys.path.insert(0, os.path.expanduser("~/Code/cluster-notify"))
        from notify import training_complete
        summary = " / ".join("{}: mask mAP50={}".format(r["variant"], r["mask_map50"]) for r in rows)
        training_complete("scrap", "exp2 컷오프 스윕 완료", summary)
    except Exception:
        pass


if __name__ == "__main__":
    main()
