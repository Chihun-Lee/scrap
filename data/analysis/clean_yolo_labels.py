"""
YOLO seg 라벨 손상 polygon 검증 및 정리
=========================================
TAL assigner shape mismatch (tal.py:195/200) 의 근본 원인:
- 매우 작은 polygon (degenerate) → affine 변환 시 NaN
- 점 < 3개 polygon
- 좌표가 [0,1] 범위 밖인 polygon
- 모든 점이 동일한 polygon (zero area)

YOLO seg format: <class> <x1> <y1> <x2> <y2> ... <xn> <yn> (정규화 [0,1])

각 라벨 파일을 검증하고 손상된 라인만 제거 (백업 후).
"""

import shutil
from pathlib import Path
import math

ROOT = Path(__file__).resolve().parent.parent
LABELS_DIRS = [
    ROOT / "datasets" / "labels" / "train",
    ROOT / "datasets" / "labels" / "val",
]
BACKUP_DIR = ROOT / "datasets" / "labels_backup"

# 최소 polygon 크기 (정규화 좌표 기준) — 0.005 = 640px 기준 약 3px
MIN_NORM_SIZE = 0.005
MIN_POINTS = 3


def is_valid_polygon(parts):
    """
    parts: [class, x1, y1, x2, y2, ...]
    Returns (is_valid, reason)
    """
    if len(parts) < 1 + 2 * MIN_POINTS:
        return False, f"too_few_points ({(len(parts)-1)//2})"

    try:
        cls = int(parts[0])
        coords = [float(x) for x in parts[1:]]
    except ValueError:
        return False, "parse_error"

    if len(coords) % 2 != 0:
        return False, "odd_coords"

    # NaN/Inf 검사
    for c in coords:
        if not math.isfinite(c):
            return False, "nan_or_inf"
        if c < 0.0 or c > 1.0:
            return False, f"out_of_range ({c:.3f})"

    # bbox 계산
    xs = coords[0::2]
    ys = coords[1::2]
    bw = max(xs) - min(xs)
    bh = max(ys) - min(ys)

    if bw < MIN_NORM_SIZE or bh < MIN_NORM_SIZE:
        return False, f"degenerate ({bw*640:.1f}x{bh*640:.1f}px)"

    # 모든 점이 동일한지
    unique_pts = set(zip(xs, ys))
    if len(unique_pts) < MIN_POINTS:
        return False, f"only_{len(unique_pts)}_unique_points"

    return True, "ok"


def process_label_file(label_path: Path):
    """라벨 파일 검증 및 정리. 반환: (kept, removed, reasons)"""
    with open(label_path, "r") as f:
        lines = f.readlines()

    kept_lines = []
    removed = []

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        valid, reason = is_valid_polygon(parts)
        if valid:
            kept_lines.append(line + "\n")
        else:
            removed.append((i, reason))

    return kept_lines, removed


def main():
    print("=" * 60)
    print("  YOLO Label Sanitizer")
    print("=" * 60)

    # 백업
    if not BACKUP_DIR.exists():
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        for d in LABELS_DIRS:
            split = d.name
            backup = BACKUP_DIR / split
            print(f"\n[BACKUP] {d} → {backup}")
            shutil.copytree(d, backup)
    else:
        print(f"\n[INFO] 백업 이미 존재: {BACKUP_DIR}")

    # 검증 + 정리
    total_files = 0
    total_lines = 0
    total_removed = 0
    files_modified = 0
    files_emptied = 0
    reason_counter = {}

    for d in LABELS_DIRS:
        print(f"\n── {d.name} ──")
        label_files = sorted(d.glob("*.txt"))
        for lp in label_files:
            total_files += 1
            kept, removed = process_label_file(lp)
            total_lines += len(kept) + len(removed)
            total_removed += len(removed)

            if removed:
                files_modified += 1
                for _, reason in removed:
                    reason_counter[reason] = reason_counter.get(reason, 0) + 1
                # 정리된 내용 저장
                with open(lp, "w") as f:
                    f.writelines(kept)
                if not kept:
                    files_emptied += 1

    print(f"\n{'='*60}")
    print(f"  요약")
    print(f"{'='*60}")
    print(f"  검사 파일: {total_files}")
    print(f"  검사 라인: {total_lines}")
    print(f"  제거 라인: {total_removed} ({total_removed/max(total_lines,1)*100:.2f}%)")
    print(f"  수정 파일: {files_modified}")
    print(f"  비워진 파일: {files_emptied}")
    print(f"\n  제거 사유 (top 20):")
    sorted_reasons = sorted(reason_counter.items(), key=lambda x: -x[1])
    for reason, cnt in sorted_reasons[:20]:
        print(f"    {reason}: {cnt}")


if __name__ == "__main__":
    main()
