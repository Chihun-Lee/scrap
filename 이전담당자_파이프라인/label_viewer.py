"""
Steel Scrap Label Viewer — Tkinter GUI

4K LabelMe 폴리곤을 이미지 위에 올려 확인하는 데스크톱 뷰어.
Stage 4종 지원:
  - original   : datasets/{split}_data           (89 raw classes)
  - filtered   : datasets/{split}_data_filtered  (8px 필터 후)
  - remapped   : datasets/{split}_remapped       (19 merged classes)
  - yolo       : datasets/labels/{split}         (YOLO seg txt, 학습 입력)

Split: train / val
기능:
  - 파일 리스트 검색, 클릭/화살표 탐색
  - 클래스별 색상 오버레이, 클래스 토글 on/off
  - 인스턴스 수/클래스 분포 사이드 패널
  - matplotlib 툴바로 확대/이동/저장

실행:
    conda activate scrap
    python tools/label_viewer.py
"""

from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)
from matplotlib.patches import Polygon
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS = PROJECT_ROOT / "datasets"

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def _stage_dirs(stage: str, split: str) -> tuple[Path, Path, str]:
    """Return (image_dir, label_dir, label_kind) for stage/split.

    label_kind ∈ {"labelme", "yolo"}.
    """
    if stage == "original":
        d = DATASETS / f"{split}_data"
        return d, d, "labelme"
    if stage == "filtered":
        d = DATASETS / f"{split}_data_filtered"
        return d, d, "labelme"
    if stage == "remapped":
        return DATASETS / "images" / split, DATASETS / f"{split}_remapped", "labelme"
    if stage == "yolo":
        return DATASETS / "images" / split, DATASETS / "labels" / split, "yolo"
    raise ValueError(stage)


def _load_yolo_classes() -> list[str]:
    f = DATASETS / "classes.txt"
    if f.exists():
        return [line.strip() for line in f.read_text().splitlines() if line.strip()]
    return []


YOLO_CLASSES = _load_yolo_classes()

_COLOR_CACHE: dict[str, tuple[float, float, float, float]] = {}


def get_color(label: str) -> tuple[float, float, float, float]:
    if label in _COLOR_CACHE:
        return _COLOR_CACHE[label]
    cmap = plt.get_cmap("tab20")
    idx = abs(hash(label)) % 20
    c = cmap(idx)
    _COLOR_CACHE[label] = c
    return c


def _find_image(label_path: Path, image_dir: Path, image_ref: str | None) -> Path | None:
    if image_ref:
        cand = (label_path.parent / image_ref).resolve()
        if cand.exists():
            return cand
    stem = label_path.stem
    for base in (image_dir, label_path.parent):
        for ext in IMAGE_EXTS:
            cand = base / f"{stem}{ext}"
            if cand.exists():
                return cand
    return None


def _parse_labelme(path: Path) -> tuple[list[dict], str | None]:
    """Return (shapes, image_ref). shapes items: {label, points: list[[x,y]]}."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    shapes = []
    for s in data.get("shapes", []):
        lbl = s.get("label", "")
        pts = s.get("points", [])
        if not lbl or len(pts) < 3:
            continue
        shapes.append({"label": lbl, "points": pts})
    return shapes, data.get("imagePath")


def _parse_yolo(path: Path, img_w: int, img_h: int) -> list[dict]:
    """Parse YOLO seg txt → shapes list in absolute pixel coords."""
    shapes = []
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        cls_id = int(parts[0])
        lbl = YOLO_CLASSES[cls_id] if 0 <= cls_id < len(YOLO_CLASSES) else f"class_{cls_id}"
        coords = [float(v) for v in parts[1:]]
        pts = [[coords[i] * img_w, coords[i + 1] * img_h] for i in range(0, len(coords) - 1, 2)]
        if len(pts) < 3:
            continue
        shapes.append({"label": lbl, "points": pts})
    return shapes


class ViewerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Steel Scrap Label Viewer")
        self.root.geometry("1700x950")

        self.stage_var = tk.StringVar(value="original")
        self.split_var = tk.StringVar(value="train")
        self.search_var = tk.StringVar()
        self.info_var = tk.StringVar()
        self.show_fill_var = tk.BooleanVar(value=True)
        self.show_labels_var = tk.BooleanVar(value=False)

        # Per-class visibility (persists across images for a stage session)
        self.class_vars: dict[str, tk.BooleanVar] = {}

        self.all_files: list[str] = []
        self.filtered_files: list[str] = []
        self.current_label_path: Path | None = None
        self.current_image_path: Path | None = None
        self.current_shapes: list[dict] = []
        self._img_arr: np.ndarray | None = None

        self._build_ui()
        self._refresh_file_list()

    # ── UI ─────────────────────────────────────────────
    def _build_ui(self):
        top = ttk.Frame(self.root, padding=6)
        top.pack(fill=tk.X)

        ttk.Label(top, text="Stage:").pack(side=tk.LEFT)
        stage_box = ttk.Combobox(
            top, textvariable=self.stage_var, state="readonly", width=10,
            values=["original", "filtered", "remapped", "yolo"],
        )
        stage_box.pack(side=tk.LEFT, padx=(4, 12))
        stage_box.bind("<<ComboboxSelected>>", lambda e: self._refresh_file_list())

        ttk.Label(top, text="Split:").pack(side=tk.LEFT)
        split_box = ttk.Combobox(
            top, textvariable=self.split_var, state="readonly", width=6,
            values=["train", "val"],
        )
        split_box.pack(side=tk.LEFT, padx=(4, 12))
        split_box.bind("<<ComboboxSelected>>", lambda e: self._refresh_file_list())

        ttk.Label(top, text="Search:").pack(side=tk.LEFT)
        entry = ttk.Entry(top, textvariable=self.search_var, width=28)
        entry.pack(side=tk.LEFT, padx=4)
        self.search_var.trace_add("write", lambda *_: self._apply_search())

        ttk.Checkbutton(top, text="Fill", variable=self.show_fill_var,
                        command=self._render).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Checkbutton(top, text="Labels", variable=self.show_labels_var,
                        command=self._render).pack(side=tk.LEFT)

        ttk.Label(top, textvariable=self.info_var, foreground="#555").pack(side=tk.RIGHT)

        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # Left: file list
        left = ttk.Frame(paned)
        paned.add(left, weight=1)
        self.file_list = tk.Listbox(left, activestyle="dotbox", exportselection=False)
        self.file_list.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        sb = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self.file_list.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.file_list.config(yscrollcommand=sb.set)
        self.file_list.bind("<<ListboxSelect>>", lambda e: self._on_select_file())

        self.root.bind("<Up>", lambda e: self._step(-1))
        self.root.bind("<Down>", lambda e: self._step(1))
        self.root.bind("<Prior>", lambda e: self._step(-10))
        self.root.bind("<Next>", lambda e: self._step(10))

        # Center: matplotlib canvas
        center = ttk.Frame(paned)
        paned.add(center, weight=5)
        self.fig, self.ax = plt.subplots(figsize=(12, 7))
        self.fig.patch.set_facecolor("#f0f0f0")
        self.canvas = FigureCanvasTkAgg(self.fig, master=center)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas, center).update()

        # Right: classes + stats
        right = ttk.Frame(paned)
        paned.add(right, weight=1)

        cls_head = ttk.Frame(right)
        cls_head.pack(fill=tk.X, padx=4, pady=(6, 0))
        ttk.Label(cls_head, text="Classes", font=("", 11, "bold")).pack(side=tk.LEFT)
        ttk.Button(cls_head, text="All", width=4,
                   command=lambda: self._set_all_classes(True)).pack(side=tk.RIGHT, padx=2)
        ttk.Button(cls_head, text="None", width=5,
                   command=lambda: self._set_all_classes(False)).pack(side=tk.RIGHT)

        cls_frame_wrap = ttk.Frame(right)
        cls_frame_wrap.pack(fill=tk.BOTH, expand=True, padx=4)
        self.class_canvas = tk.Canvas(cls_frame_wrap, highlightthickness=0)
        self.class_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        cls_sb = ttk.Scrollbar(cls_frame_wrap, orient=tk.VERTICAL,
                               command=self.class_canvas.yview)
        cls_sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.class_canvas.config(yscrollcommand=cls_sb.set)
        self.class_frame = ttk.Frame(self.class_canvas)
        self.class_canvas.create_window((0, 0), window=self.class_frame, anchor="nw")
        self.class_frame.bind(
            "<Configure>",
            lambda e: self.class_canvas.configure(scrollregion=self.class_canvas.bbox("all")),
        )

        ttk.Separator(right, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)
        ttk.Label(right, text="Stats", font=("", 11, "bold")).pack(anchor=tk.W, padx=4)
        self.stats_text = tk.Text(right, height=14, width=34,
                                  state=tk.DISABLED, wrap=tk.NONE)
        self.stats_text.pack(fill=tk.BOTH, expand=False, padx=4, pady=4)

    # ── File list ──────────────────────────────────────
    def _label_glob_pattern(self) -> str:
        return "*.txt" if self.stage_var.get() == "yolo" else "*.json"

    def _refresh_file_list(self):
        _, label_dir, _ = _stage_dirs(self.stage_var.get(), self.split_var.get())
        if not label_dir.exists():
            self.all_files = []
            self.info_var.set(f"[missing] {label_dir}")
        else:
            pat = self._label_glob_pattern()
            self.all_files = sorted(p.name for p in label_dir.glob(pat))
            self.info_var.set(f"{len(self.all_files)} files | {label_dir.relative_to(PROJECT_ROOT)}")
        self._apply_search()

    def _apply_search(self):
        q = self.search_var.get().lower()
        self.filtered_files = [f for f in self.all_files if q in f.lower()]
        self.file_list.delete(0, tk.END)
        for f in self.filtered_files:
            self.file_list.insert(tk.END, f)
        if self.filtered_files:
            self.file_list.selection_clear(0, tk.END)
            self.file_list.selection_set(0)
            self.file_list.see(0)
            self._on_select_file()
        else:
            self._clear_view()

    def _step(self, delta: int):
        sel = self.file_list.curselection()
        if not sel or not self.filtered_files:
            return
        idx = max(0, min(len(self.filtered_files) - 1, sel[0] + delta))
        self.file_list.selection_clear(0, tk.END)
        self.file_list.selection_set(idx)
        self.file_list.see(idx)
        self._on_select_file()

    def _clear_view(self):
        self.ax.clear()
        self.ax.set_facecolor("#e8e8e8")
        self.ax.set_xticks([]); self.ax.set_yticks([])
        self.canvas.draw_idle()
        self._write_stats("(no file)")

    # ── File load ──────────────────────────────────────
    def _on_select_file(self):
        sel = self.file_list.curselection()
        if not sel:
            return
        fname = self.file_list.get(sel[0])
        stage = self.stage_var.get()
        split = self.split_var.get()
        image_dir, label_dir, kind = _stage_dirs(stage, split)
        label_path = label_dir / fname

        try:
            if kind == "labelme":
                shapes, image_ref = _parse_labelme(label_path)
                image_path = _find_image(label_path, image_dir, image_ref)
            else:
                image_path = _find_image(label_path, image_dir, None)
                if image_path is None:
                    shapes = []
                else:
                    with Image.open(image_path) as im:
                        w, h = im.size
                    shapes = _parse_yolo(label_path, w, h)
        except Exception as e:
            messagebox.showerror("Load error", f"{label_path.name}\n{e}")
            return

        self.current_label_path = label_path
        self.current_image_path = image_path
        self.current_shapes = shapes
        self._img_arr = np.asarray(Image.open(image_path)) if image_path else None

        self._rebuild_class_toggles()
        self._render()
        self._write_stats_for_current()

    # ── Class toggles ──────────────────────────────────
    def _rebuild_class_toggles(self):
        labels = sorted({s["label"] for s in self.current_shapes})
        # Reuse BooleanVars when names match (preserves user toggle state)
        for child in self.class_frame.winfo_children():
            child.destroy()
        for lbl in labels:
            var = self.class_vars.setdefault(lbl, tk.BooleanVar(value=True))
            color = get_color(lbl)
            hex_color = "#{:02x}{:02x}{:02x}".format(
                int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
            )
            row = ttk.Frame(self.class_frame)
            row.pack(fill=tk.X, anchor=tk.W)
            tk.Label(row, text="   ", bg=hex_color).pack(side=tk.LEFT, padx=(0, 4))
            ttk.Checkbutton(row, text=lbl, variable=var,
                            command=self._render).pack(side=tk.LEFT, anchor=tk.W)

    def _set_all_classes(self, value: bool):
        for v in self.class_vars.values():
            v.set(value)
        self._render()

    # ── Rendering ──────────────────────────────────────
    def _render(self):
        self.ax.clear()
        self.ax.set_facecolor("#e8e8e8")
        if self._img_arr is not None:
            self.ax.imshow(self._img_arr)
        else:
            self.ax.text(0.5, 0.5, "(image not found)", ha="center", va="center",
                         transform=self.ax.transAxes)

        fill = self.show_fill_var.get()
        show_labels = self.show_labels_var.get()
        for shape in self.current_shapes:
            lbl = shape["label"]
            var = self.class_vars.get(lbl)
            if var is not None and not var.get():
                continue
            color = get_color(lbl)
            pts = shape["points"]
            face = (*color[:3], 0.30) if fill else (0, 0, 0, 0)
            poly = Polygon(pts, closed=True, fill=fill,
                           facecolor=face,
                           edgecolor=(*color[:3], 0.95),
                           linewidth=1.0)
            self.ax.add_patch(poly)
            if show_labels:
                cx = sum(p[0] for p in pts) / len(pts)
                cy = sum(p[1] for p in pts) / len(pts)
                self.ax.text(cx, cy, lbl, fontsize=6, color="white",
                             ha="center", va="center",
                             bbox=dict(facecolor=color[:3], alpha=0.7, pad=0.8,
                                       edgecolor="none"))

        self.ax.set_xticks([]); self.ax.set_yticks([])
        name = self.current_label_path.name if self.current_label_path else ""
        self.ax.set_title(name)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    # ── Stats ──────────────────────────────────────────
    def _write_stats_for_current(self):
        if not self.current_shapes:
            self._write_stats("(empty labels)")
            return
        counts: dict[str, int] = {}
        for s in self.current_shapes:
            counts[s["label"]] = counts.get(s["label"], 0) + 1
        total = sum(counts.values())
        lines = [
            f"file : {self.current_label_path.name if self.current_label_path else '-'}",
            f"image: {self.current_image_path.name if self.current_image_path else '(missing)'}",
            f"instances: {total}",
            f"classes  : {len(counts)}",
            "",
            f"{'count':>6}  label",
            "-" * 32,
        ]
        for lbl, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"{n:6d}  {lbl}")
        self._write_stats("\n".join(lines))

    def _write_stats(self, text: str):
        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete("1.0", tk.END)
        self.stats_text.insert("1.0", text)
        self.stats_text.config(state=tk.DISABLED)


def main():
    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam")
    except Exception:
        pass
    ViewerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
