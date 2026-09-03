from __future__ import annotations

import csv
import os
import queue
import re
import subprocess
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageGrab, ImageOps, ImageTk
from tkinterdnd2 import DND_FILES, TkinterDnD

from engine import (
    IMAGE_EXTENSIONS,
    EmbeddingEngine,
    LibraryIndex,
    Match,
    annotation_lines,
    annotate_result,
    app_data_dir,
    create_overview,
    detect_color_name,
    detect_red_text_tokens,
    iter_images,
    load_settings,
    next_task_name,
    parse_size_tokens,
    parse_marked_color_text,
    save_settings,
)


APP_TITLE = "服装找款与配货标注工具 V8｜对比界面版"


@dataclass
class WorkItem:
    input_path: Path
    color: str = ""
    sizes_text: str = ""
    candidates: list[Match] = field(default_factory=list)
    candidate_limit: int = 20
    selected: Path | None = None
    decision_stage: str = ""
    status: str = "待分析"
    output_unmatched: bool = False
    analyzed: bool = False


class ClothingMatcherApp(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        width = min(1400, max(760, screen_w - 40))
        height = min(850, max(520, screen_h - 90))
        self.geometry(f"{width}x{height}")
        self.minsize(700, 480)
        self.resizable(True, True)

        self.settings = load_settings()
        models_dir = Path(__file__).resolve().parent / "models"
        model_path = models_dir / "sscd_disc_mixup.onnx"
        if not model_path.exists():
            model_path = models_dir / "dinov2_small_q4.onnx"
        if not model_path.exists():
            model_path = models_dir / "vision_model_int8.onnx"
        self.engine = EmbeddingEngine(model_path)
        self.index: LibraryIndex | None = None
        self.items: list[WorkItem] = []
        self.current_index: int | None = None
        self._selecting_item = False
        self._closing = False
        self._poll_after_id = None
        self._startup_after_ids: list[str] = []
        self.events: queue.Queue = queue.Queue()
        self.input_photo = None
        self.candidate_photo = None
        # Each preview keeps only its currently displayed image (capped while
        # loading). Zooming/panning is therefore purely a UI operation; it does
        # not rerun OCR, the visual model, or the candidate search.
        self.input_view = None
        self.candidate_view = None
        self._preview_pan_start: dict[str, tuple[int, int]] = {}

        self.library_var = tk.StringVar(value=self.settings.get("library", ""))
        self.output_var = tk.StringVar(value=self.settings.get("output", ""))
        self.color_options = ["黑色", "白色", "灰色", "卡其"]
        self.color_choice_var = tk.StringVar()
        self.color_other_var = tk.StringVar()
        self.size_order = ["M", "L", "XL", "2XL", "3XL"]
        self.size_vars = {size: tk.StringVar() for size in self.size_order}
        self.status_var = tk.StringVar(value=f"匹配引擎：{self.engine.mode}")
        self.summary_var = tk.StringVar(value="尚未导入照片")
        self.current_var = tk.StringVar(value="")
        self.candidate_header_var = tk.StringVar(value="候选款式")
        self.auto_next_var = tk.BooleanVar(value=True)

        self._build_ui()
        self._startup_after_ids.append(self.after(250, self._set_default_pane_sizes))
        self.bind("<F5>", lambda _event: self._analyze_all())
        self.bind("<Control-Return>", self._confirm_candidate)
        self.bind("<Control-Right>", lambda _event: self._select_next_pending())
        self.bind("<Control-Left>", lambda _event: self._select_previous_item())
        self.bind("<Control-e>", lambda _event: self._export_all())
        self._startup_after_ids.append(self.after(200, self._enable_safe_file_drop))
        self._poll_after_id = self.after(100, self._poll_events)
        if self.library_var.get():
            self._startup_after_ids.append(self.after(300, self._load_or_build_index))

    def _build_ui(self):
        outer = ttk.Frame(self)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)
        self.page_canvas = tk.Canvas(outer, highlightthickness=0)
        vertical = tk.Scrollbar(
            outer, orient=tk.VERTICAL, command=self.page_canvas.yview,
            width=28, troughcolor="#c8c8c8", relief=tk.RAISED,
        )
        horizontal = tk.Scrollbar(
            outer, orient=tk.HORIZONTAL, command=self.page_canvas.xview,
            width=24, troughcolor="#c8c8c8", relief=tk.RAISED,
        )
        self.page_canvas.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        self.page_canvas.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")

        # Keep the primary output action permanently visible. It must never be
        # pushed below the viewport by image previews or high-DPI scaling.
        fixed_footer = ttk.Frame(outer, padding=(8, 6))
        fixed_footer.grid(row=2, column=0, columnspan=2, sticky="ew")
        self.export_button = ttk.Button(
            fixed_footer,
            text="导出本批全部结果  Ctrl+E",
            command=self._export_all,
        )
        self.export_button.pack(side=tk.RIGHT, ipadx=18, ipady=3)
        ttk.Label(
            fixed_footer, textvariable=self.status_var, width=42, anchor="e"
        ).pack(side=tk.RIGHT, padx=(8, 12))
        self.progress = ttk.Progressbar(fixed_footer, mode="determinate")
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True)

        page = ttk.Frame(self.page_canvas)
        self.page = page
        self._page_window = self.page_canvas.create_window((0, 0), window=page, anchor="nw")
        page.columnconfigure(0, weight=1)
        page.rowconfigure(1, weight=1)
        page.bind("<Configure>", self._refresh_page_scrollregion)
        self.page_canvas.bind("<Configure>", self._resize_scrollable_page)
        self.page_canvas.bind_all("<MouseWheel>", self._on_page_mousewheel, add="+")
        self.page_canvas.bind_all("<Shift-MouseWheel>", self._on_page_shift_mousewheel, add="+")

        top = ttk.Frame(page, padding=8)
        top.grid(row=0, column=0, sticky="ew")
        ttk.Label(top, text="款式图库").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.library_var).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(top, text="选择", command=self._choose_library).grid(row=0, column=2)
        ttk.Button(top, text="建立/更新索引", command=self._build_index_async).grid(row=0, column=3, padx=(6, 0))
        ttk.Label(top, text="结果目录").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(top, textvariable=self.output_var).grid(row=1, column=1, sticky="ew", padx=6, pady=(6, 0))
        ttk.Button(top, text="选择", command=self._choose_output).grid(row=1, column=2, pady=(6, 0))
        ttk.Button(top, text="导入实物照片", command=self._import_inputs).grid(row=1, column=3, padx=(6, 0), pady=(6, 0))
        top.columnconfigure(1, weight=1)

        main = tk.PanedWindow(
            page,
            orient=tk.HORIZONTAL,
            sashwidth=9,
            sashrelief=tk.RAISED,
            showhandle=True,
            handlesize=14,
            handlepad=18,
            sashcursor="sb_h_double_arrow",
            opaqueresize=True,
            bd=0,
            bg="#aab3bb",
        )
        self.main_panes = main
        main.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))

        left = ttk.Frame(main, padding=6)
        center = ttk.Frame(main, padding=6)
        candidate_preview = ttk.Frame(main, padding=6)
        controls = ttk.Frame(main, padding=6)
        main.add(left, minsize=190, stretch="always")
        main.add(center, minsize=280, stretch="always")
        main.add(candidate_preview, minsize=280, stretch="always")
        main.add(controls, minsize=340, stretch="always")

        ttk.Label(left, text="待处理照片").pack(anchor="w")
        ttk.Label(left, textvariable=self.summary_var, foreground="#245b8a").pack(anchor="w", pady=(2, 0))
        input_list_frame = ttk.Frame(left)
        input_list_frame.pack(fill=tk.BOTH, expand=True, pady=6)
        self.input_list = tk.Listbox(input_list_frame, exportselection=False)
        input_scroll = ttk.Scrollbar(
            input_list_frame, orient=tk.VERTICAL, command=self.input_list.yview
        )
        self.input_list.configure(yscrollcommand=input_scroll.set)
        self.input_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        input_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.input_list.bind("<<ListboxSelect>>", self._on_item_select)
        self.input_list.bind("<Control-v>", self._paste_inputs)
        self.input_list.bind("<Control-V>", self._paste_inputs)
        self.input_list.bind("<Button-3>", self._show_input_menu)
        row = ttk.Frame(left)
        row.pack(fill=tk.X)
        ttk.Button(row, text="分析新照片  F5", command=self._analyze_all).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(row, text="移除", command=self._remove_current).pack(side=tk.LEFT, padx=(6, 0))
        nav = ttk.Frame(left)
        nav.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(nav, text="← 上一张", command=self._select_previous_item).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(nav, text="下一张待确认 →", command=self._select_next_pending).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))
        ttk.Button(left, text="导入整个照片文件夹", command=self._import_input_folder).pack(fill=tk.X, pady=(6, 0))

        center_header = ttk.Frame(center)
        center_header.pack(fill=tk.X)
        ttk.Label(center_header, text="实物照片").pack(side=tk.LEFT)
        ttk.Label(center_header, textvariable=self.current_var, foreground="#245b8a").pack(side=tk.RIGHT)
        ttk.Button(
            center_header, text="适应窗口", command=lambda: self._reset_preview("input_photo")
        ).pack(side=tk.RIGHT, padx=(0, 8))
        ttk.Label(center_header, text="↔ 可调宽", foreground="#687681").pack(side=tk.RIGHT, padx=(0, 8))
        self.input_canvas = tk.Canvas(
            center, bg="#d8d8d8", width=1, height=1, highlightthickness=0
        )
        self.input_canvas.pack(fill=tk.BOTH, expand=True, pady=(6, 8))
        self._setup_preview_canvas(self.input_canvas, "input_photo")

        preview_header = ttk.Frame(candidate_preview)
        preview_header.pack(fill=tk.X)
        ttk.Label(preview_header, text="候选/结果预览").pack(side=tk.LEFT)
        ttk.Button(
            preview_header, text="适应窗口", command=lambda: self._reset_preview("candidate_photo")
        ).pack(side=tk.RIGHT)
        ttk.Label(preview_header, text="↔ 可调宽", foreground="#687681").pack(side=tk.RIGHT, padx=(0, 8))
        self.candidate_canvas = tk.Canvas(
            candidate_preview, bg="#d8d8d8", width=1, height=1, highlightthickness=0
        )
        self.candidate_canvas.pack(fill=tk.BOTH, expand=True, pady=(6, 8))
        self._setup_preview_canvas(self.candidate_canvas, "candidate_photo")

        candidate_header = ttk.Frame(controls)
        candidate_header.pack(fill=tk.X)
        ttk.Label(candidate_header, textvariable=self.candidate_header_var).pack(side=tk.LEFT, anchor="w")
        self.more_candidates_button = ttk.Button(
            candidate_header,
            text="再显示20款",
            command=self._show_more_candidates,
        )
        self.more_candidates_button.pack(side=tk.RIGHT)
        candidate_options = ttk.Frame(controls)
        candidate_options.pack(fill=tk.X, pady=(3, 0))
        self.candidate_options = candidate_options
        self.auto_next_check = ttk.Checkbutton(
            candidate_options,
            text="确认后到下一张",
            variable=self.auto_next_var,
        )
        self.auto_next_check.pack(side=tk.LEFT)
        candidate_list_frame = ttk.Frame(controls)
        candidate_list_frame.pack(fill=tk.BOTH, expand=True, pady=6)
        self.candidate_list = tk.Listbox(
            candidate_list_frame, exportselection=False, height=20
        )
        candidate_scroll = ttk.Scrollbar(
            candidate_list_frame, orient=tk.VERTICAL, command=self.candidate_list.yview
        )
        self.candidate_list.configure(yscrollcommand=candidate_scroll.set)
        self.candidate_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        candidate_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.candidate_list.bind("<<ListboxSelect>>", self._on_candidate_select)
        self.candidate_list.bind("<Double-Button-1>", self._confirm_candidate)
        btns = ttk.Frame(controls)
        self.candidate_actions = btns
        # Primary actions stay above the preview, so a loaded portrait photo can
        # never push the buttons off a smaller screen.
        btns.pack(fill=tk.X, pady=(0, 6))
        ttk.Button(btns, text="确认候选  Ctrl+Enter", command=self._confirm_candidate).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(btns, text="下一张 →", command=self._select_next_pending).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))
        ttk.Button(btns, text="手动选择款式图", command=self._manual_select).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))
        ttk.Button(btns, text="未匹配", command=self._mark_unmatched).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))
        # Put the fields immediately under the confirmation controls.  The
        # normal workflow is now candidate → color/quantity → confirm, without
        # moving the mouse across panes.
        self._build_annotation_form(controls)

    def _build_annotation_form(self, parent: tk.Misc):
        """Build a wrapped editor that remains complete in a narrow control rail."""
        form = ttk.LabelFrame(parent, text="标注内容（回车预览）", padding=(6, 4))
        form.pack(fill=tk.X, pady=(0, 2))
        ttk.Label(form, text="颜色").grid(row=0, column=0, sticky="w")
        color_frame = ttk.Frame(form)
        color_frame.grid(row=0, column=1, sticky="ew", padx=6)
        self.color_frame = color_frame
        for option in self.color_options:
            ttk.Radiobutton(
                color_frame,
                text=option,
                value=option,
                variable=self.color_choice_var,
            ).pack(side=tk.LEFT, padx=(0, 7))
        color_other_frame = ttk.Frame(form)
        color_other_frame.grid(row=1, column=1, sticky="ew", padx=6, pady=(3, 0))
        self.color_other_frame = color_other_frame
        ttk.Radiobutton(
            color_other_frame,
            text="其他文字",
            value="其他",
            variable=self.color_choice_var,
        ).pack(side=tk.LEFT)
        self.color_other_entry = ttk.Entry(
            color_other_frame, textvariable=self.color_other_var, width=11
        )
        self.color_other_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))
        self.color_other_entry.bind(
            "<KeyPress>", lambda _event: self.color_choice_var.set("其他")
        )
        ttk.Label(form, text="尺码数量").grid(row=2, column=0, sticky="nw", pady=(5, 0))
        size_frame = ttk.Frame(form)
        size_frame.grid(row=2, column=1, sticky="ew", padx=6, pady=(5, 0))
        self.size_frame = size_frame
        for index, size in enumerate(self.size_order):
            row, col = divmod(index, 3)
            ttk.Label(size_frame, text=f"{size}:").grid(row=row, column=col * 2, sticky="w", pady=(0, 3))
            entry = ttk.Entry(size_frame, textvariable=self.size_vars[size], width=5)
            entry.grid(row=row, column=col * 2 + 1, sticky="w", padx=(2, 10), pady=(0, 3))
            entry.bind("<Return>", self._on_sizes_return)
        ttk.Label(form, text="其他文字").grid(row=3, column=0, sticky="nw", pady=(5, 0))
        self.extra_textbox = tk.Text(form, height=2, width=34, wrap="word", undo=True)
        self.extra_textbox.grid(row=3, column=1, sticky="nsew", padx=6, pady=(5, 0))
        self.extra_textbox.bind("<Return>", self._on_sizes_return)
        form.columnconfigure(1, weight=1)

    def _setup_preview_canvas(self, canvas: tk.Canvas, attr: str):
        canvas.bind("<Configure>", lambda _event, key=attr: self._draw_preview(key), add="+")
        canvas.bind("<MouseWheel>", lambda event, key=attr: self._on_preview_zoom(event, key))
        canvas.bind("<Button-4>", lambda event, key=attr: self._on_preview_zoom(event, key, 120))
        canvas.bind("<Button-5>", lambda event, key=attr: self._on_preview_zoom(event, key, -120))
        canvas.bind("<ButtonPress-1>", lambda event, key=attr: self._start_preview_pan(event, key))
        canvas.bind("<B1-Motion>", lambda event, key=attr: self._move_preview_pan(event, key))
        canvas.bind("<Double-Button-1>", lambda _event, key=attr: self._reset_preview(key))
        canvas.configure(cursor="fleur")

    def _preview_canvas(self, attr: str) -> tk.Canvas:
        return self.input_canvas if attr == "input_photo" else self.candidate_canvas

    def _preview_view(self, attr: str):
        return self.input_view if attr == "input_photo" else self.candidate_view

    def _set_preview_view(self, attr: str, view):
        if attr == "input_photo":
            self.input_view = view
        else:
            self.candidate_view = view

    def _draw_preview(self, attr: str):
        canvas = self._preview_canvas(attr)
        view = self._preview_view(attr)
        if not view:
            canvas.delete("all")
            return
        source = view["image"]
        canvas.update_idletasks()
        available_w = max(80, canvas.winfo_width() - 12)
        available_h = max(80, canvas.winfo_height() - 12)
        fit_scale = min(available_w / source.width, available_h / source.height, 1.0)
        view["fit_scale"] = fit_scale
        scale = fit_scale * view["zoom"]
        display_size = (
            max(1, round(source.width * scale)),
            max(1, round(source.height * scale)),
        )
        rendered = source.resize(display_size, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(rendered)
        setattr(self, attr, photo)
        x = canvas.winfo_width() // 2 + round(view["offset_x"])
        y = canvas.winfo_height() // 2 + round(view["offset_y"])
        preview_items = canvas.find_withtag("preview")
        if preview_items:
            canvas.itemconfigure(preview_items[0], image=photo)
            canvas.coords(preview_items[0], x, y)
        else:
            canvas.create_image(x, y, image=photo, anchor="center", tags="preview")

    def _on_preview_zoom(self, event, attr: str, wheel_delta: int | None = None):
        view = self._preview_view(attr)
        if not view:
            return "break"
        canvas = self._preview_canvas(attr)
        old_scale = max(1e-6, view.get("fit_scale", 1.0) * view["zoom"])
        delta = wheel_delta if wheel_delta is not None else event.delta
        factor = 1.20 if delta > 0 else 1 / 1.20
        new_zoom = max(0.45, min(5.0, view["zoom"] * factor))
        if new_zoom == view["zoom"]:
            return "break"
        new_scale = max(1e-6, view.get("fit_scale", 1.0) * new_zoom)
        center_x, center_y = canvas.winfo_width() / 2, canvas.winfo_height() / 2
        mouse_x, mouse_y = event.x, event.y
        # Keep the detail under the pointer in place while zooming.
        view["offset_x"] = mouse_x - center_x - (mouse_x - center_x - view["offset_x"]) * new_scale / old_scale
        view["offset_y"] = mouse_y - center_y - (mouse_y - center_y - view["offset_y"]) * new_scale / old_scale
        view["zoom"] = new_zoom
        self._draw_preview(attr)
        return "break"

    def _start_preview_pan(self, event, attr: str):
        if self._preview_view(attr):
            self._preview_pan_start[attr] = (event.x, event.y)
        return "break"

    def _move_preview_pan(self, event, attr: str):
        view = self._preview_view(attr)
        start = self._preview_pan_start.get(attr)
        if not view or not start:
            return "break"
        dx = event.x - start[0]
        dy = event.y - start[1]
        view["offset_x"] += dx
        view["offset_y"] += dy
        self._preview_pan_start[attr] = (event.x, event.y)
        # Move the already-rendered canvas item. Rebuilding a PhotoImage for
        # every mouse-motion event caused the black-frame flicker in V8_1.
        self._preview_canvas(attr).move("preview", dx, dy)
        return "break"

    def _reset_preview(self, attr: str):
        view = self._preview_view(attr)
        if view:
            view["zoom"] = 1.0
            view["offset_x"] = 0.0
            view["offset_y"] = 0.0
            self._draw_preview(attr)

    def _set_default_pane_sizes(self):
        """Give both portrait previews more room on the first launch."""
        self.update_idletasks()
        total = self.main_panes.winfo_width()
        if total < 700:
            return
        # Compact queue, two roomy portrait previews, usable editing rail.
        for index, ratio in enumerate((0.16, 0.45, 0.75)):
            try:
                self.main_panes.sash_place(index, int(total * ratio), 0)
            except tk.TclError:
                pass

    def _refresh_page_scrollregion(self, _event=None):
        self.page_canvas.configure(scrollregion=self.page_canvas.bbox("all"))

    def _resize_scrollable_page(self, event):
        # Include the real requested size. This matters on high-DPI screens where
        # controls can be taller than their nominal design size.
        self.page.update_idletasks()
        # Let panes shrink on smaller displays. If the screen is smaller still,
        # the outer horizontal/vertical scrollbars remain available.
        width = max(event.width, 1240)
        height = max(event.height, 620)
        self.page_canvas.itemconfigure(self._page_window, width=width, height=height)
        self._refresh_page_scrollregion()

    def _on_page_mousewheel(self, event):
        if isinstance(event.widget, (tk.Listbox, tk.Text)):
            return None
        self.page_canvas.yview_scroll(int(-event.delta / 120), "units")

    def _on_page_shift_mousewheel(self, event):
        self.page_canvas.xview_scroll(int(-event.delta / 120), "units")

    def _choose_library(self):
        path = filedialog.askdirectory(title="选择款式图库文件夹")
        if path:
            self.library_var.set(path)
            if not self.output_var.get():
                self.output_var.set(str(Path(path) / "标注结果"))
            self._persist()
            self._load_or_build_index()

    def _choose_output(self):
        path = filedialog.askdirectory(title="选择结果保存文件夹")
        if path:
            self.output_var.set(path)
            self._persist()

    def _persist(self):
        save_settings({"library": self.library_var.get(), "output": self.output_var.get()})

    def _load_or_build_index(self):
        library = Path(self.library_var.get())
        if not library.is_dir():
            return
        self.index = LibraryIndex(library, self.engine)
        if self.index.load():
            self.status_var.set(f"图库已就绪：{len(self.index.paths)} 款｜{self.engine.mode}")
        else:
            self._build_index_async()

    def _build_index_async(self):
        library = Path(self.library_var.get())
        if not library.is_dir():
            messagebox.showwarning("提示", "请先选择有效的款式图库文件夹")
            return
        self.index = LibraryIndex(library, self.engine)
        self._index_rebuild_requested = True
        self.progress["value"] = 0
        self.status_var.set("正在建立图库索引…")

        def worker():
            def progress(done, total, name):
                self.events.put(("index_progress", done, total, name))
            try:
                count = self.index.build(progress)
                self.events.put(("index_done", count))
            except Exception as exc:
                self.events.put(("error", f"建立索引失败：{exc}"))

        threading.Thread(target=worker, daemon=True).start()

    def _import_inputs(self):
        paths = filedialog.askopenfilenames(
            title="选择实物衣服照片",
            filetypes=[("图片", "*.png *.jpg *.jpeg *.webp *.bmp")],
        )
        self._add_input_paths(Path(raw) for raw in paths)

    def _import_input_folder(self):
        folder = filedialog.askdirectory(title="选择实物照片文件夹")
        if folder:
            self._add_input_paths([Path(folder)])

    def _add_input_paths(self, paths):
        expanded: list[Path] = []
        for raw in paths:
            path = Path(raw)
            if path.is_dir():
                expanded.extend(iter_images(path))
            elif path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                expanded.append(path)
        added = 0
        for path in expanded:
            if not any(x.input_path == path for x in self.items):
                self.items.append(WorkItem(path))
                added += 1
        self._refresh_input_list()
        if self.items and self.current_index is None:
            self.input_list.selection_set(0)
            self._select_item(0)
        if added:
            self.status_var.set(f"已导入 {added} 张实物照片")

    def _paste_inputs(self, _event=None):
        try:
            content = ImageGrab.grabclipboard()
        except Exception:
            content = None
        if isinstance(content, list):
            self._add_input_paths(Path(path) for path in content)
            return "break"
        if isinstance(content, Image.Image):
            paste_dir = app_data_dir() / "pasted_inputs"
            paste_dir.mkdir(parents=True, exist_ok=True)
            destination = paste_dir / f"粘贴图片_{uuid.uuid4().hex[:10]}.png"
            content.convert("RGB").save(destination)
            self._add_input_paths([destination])
            return "break"
        try:
            text = self.clipboard_get().strip().strip('"')
            candidates = [Path(part.strip().strip('"')) for part in text.splitlines() if part.strip()]
            self._add_input_paths(candidates)
            if candidates:
                return "break"
        except Exception:
            pass
        messagebox.showinfo("粘贴照片", "剪贴板里没有图片文件。请先在资源管理器复制图片或文件夹。")
        return "break"

    def _show_input_menu(self, event):
        menu = tk.Menu(self, tearoff=False)
        menu.add_command(label="粘贴图片/文件夹（Ctrl+V）", command=self._paste_inputs)
        menu.add_command(label="选择多张图片", command=self._import_inputs)
        menu.add_command(label="选择整个文件夹", command=self._import_input_folder)
        menu.tk_popup(event.x_root, event.y_root)

    def _enable_safe_file_drop(self):
        """Use TkDND/OLE instead of replacing the native window procedure."""
        try:
            for target in (self, self.input_list, self.page_canvas):
                target.drop_target_register(DND_FILES)
                target.dnd_bind("<<Drop>>", self._on_safe_file_drop)
            self.status_var.set(f"可从资源管理器或微信拖入图片｜匹配引擎：{self.engine.mode}")
        except Exception as exc:
            self.status_var.set(f"拖放初始化失败，可使用 Ctrl+V 或导入按钮：{exc}")

    def _on_safe_file_drop(self, event):
        try:
            raw_items = self.tk.splitlist(event.data)
            paths = []
            for raw in raw_items:
                value = str(raw).strip().strip('"')
                if value.lower().startswith("file:///"):
                    from urllib.parse import unquote, urlparse

                    value = unquote(urlparse(value).path).lstrip("/")
                candidate = Path(value)
                if candidate.exists():
                    paths.append(candidate)
            if paths:
                self.after_idle(lambda dropped=tuple(paths): self._add_input_paths(dropped))
            else:
                self.status_var.set("拖入内容没有可读取的本地图片；可在微信中复制图片后按 Ctrl+V")
        except Exception as exc:
            # A malformed third-party drag payload must never terminate the app.
            self.status_var.set(f"这次拖入未成功，程序仍可继续使用：{exc}")
        return "copy"

    def _refresh_input_list(self):
        self.input_list.delete(0, tk.END)
        for position, item in enumerate(self.items, start=1):
            self.input_list.insert(
                tk.END,
                f"{item.status}｜图片{position:02d} {item.input_path.name}",
            )
        confirmed = sum(bool(item.selected) and not item.output_unmatched for item in self.items)
        unmatched = sum(item.output_unmatched for item in self.items)
        pending = len(self.items) - confirmed - unmatched
        self.summary_var.set(
            f"共 {len(self.items)} 张｜待确认 {pending}｜已确认 {confirmed}｜未匹配 {unmatched}"
            if self.items else "尚未导入照片"
        )

    def _on_item_select(self, _event=None):
        if self._selecting_item:
            return
        selection = self.input_list.curselection()
        if selection:
            self._save_fields(silent=True)
            self._select_item(selection[0])

    def _select_item(self, index: int):
        if not 0 <= index < len(self.items):
            return
        self.current_index = index
        item = self.items[index]
        self._selecting_item = True
        try:
            self.input_list.selection_clear(0, tk.END)
            self.input_list.selection_set(index)
            self.input_list.activate(index)
            self.input_list.see(index)
        finally:
            self._selecting_item = False
        self.current_var.set(f"第 {index + 1}/{len(self.items)} 张｜{item.status}")
        self._set_color_fields(item.color)
        self._load_size_fields(item.sizes_text)
        self._show_image(item.input_path, self.input_canvas, "input_photo")
        self._refresh_candidates(item)
        if item.selected:
            self._render_annotation_preview(item)

    def _select_previous_item(self):
        if not self.items:
            return
        current = self.current_index if self.current_index is not None else 0
        self._save_fields(silent=True)
        self._select_item((current - 1) % len(self.items))

    def _select_next_pending(self):
        if not self.items:
            return
        current = self.current_index if self.current_index is not None else -1
        for offset in range(1, len(self.items) + 1):
            index = (current + offset) % len(self.items)
            if not self.items[index].selected:
                self._save_fields(silent=True)
                self._select_item(index)
                return
        self.status_var.set("✓ 本批照片已全部确认，可以直接导出")

    def _advance_after_decision(self, decided_index: int):
        if not self.auto_next_var.get():
            return
        for offset in range(1, len(self.items) + 1):
            index = (decided_index + offset) % len(self.items)
            if not self.items[index].selected:
                self.after(120, lambda target=index: self._select_item(target))
                return

    def _set_color_fields(self, color: str):
        color = color.strip()
        if color in self.color_options:
            self.color_choice_var.set(color)
            self.color_other_var.set("")
        else:
            self.color_choice_var.set("其他")
            self.color_other_var.set("" if color == "未识别" else color)

    def _current_color(self) -> str:
        choice = self.color_choice_var.get().strip()
        if choice == "其他":
            return self.color_other_var.get().strip()
        return choice

    def _load_size_fields(self, text: str):
        for var in self.size_vars.values():
            var.set("")
        self.extra_textbox.delete("1.0", tk.END)
        extras = []
        for line in annotation_lines(text):
            parsed = parse_size_tokens(line)
            if len(parsed) == 1:
                match = re.fullmatch(r"(S|M|L|XL|2XL|3XL)(\d+)", parsed[0], re.IGNORECASE)
                if match:
                    size = match.group(1).upper()
                    self.size_vars[size].set(match.group(2))
                    continue
            extras.append(line)
        if extras:
            self.extra_textbox.insert("1.0", "\n".join(extras))

    def _show_image(self, path: Path, canvas: tk.Canvas, attr: str):
        try:
            with Image.open(path) as src:
                im = ImageOps.exif_transpose(src).convert("RGB")
                # Two preview images are retained locally for panning.  Capping
                # the longest edge keeps memory predictable even for huge phone
                # photos while still preserving more detail than the screen.
                im.thumbnail((3000, 3000), Image.Resampling.LANCZOS)
            self._set_preview_view(
                attr,
                {
                    "image": im,
                    "path": path,
                    "zoom": 1.0,
                    "offset_x": 0.0,
                    "offset_y": 0.0,
                    "fit_scale": 1.0,
                },
            )
            self._draw_preview(attr)
        except Exception:
            self._set_preview_view(attr, None)
            canvas.delete("all")

    def _analyze_all(self):
        self._save_fields()
        if not self.index or self.index.vectors is None:
            messagebox.showwarning("提示", "请先建立图库索引")
            return
        if not self.items:
            messagebox.showwarning("提示", "请先导入实物照片")
            return
        # Incremental analysis: photos already ranked but still waiting for a
        # human choice are not needlessly sent through OCR/model again when the
        # user imports a few more photos later.
        pending = [item for item in self.items if not item.selected and not item.analyzed]
        if not pending:
            self.status_var.set("没有新照片需要分析；现有候选和已确认结果都不会重复计算")
            return
        self.status_var.set(f"正在分析 {len(pending)} 张未确认照片…")
        self.progress["maximum"] = len(pending)
        self.progress["value"] = 0

        def worker():
            for i, item in enumerate(pending):
                try:
                    tokens, marked_text = detect_red_text_tokens(item.input_path)
                    # An operator-written color is authoritative. Pixel-based
                    # color estimation is only the fallback when no such text
                    # can be read from the real photo.
                    item.color = (
                        parse_marked_color_text(marked_text)
                        or detect_color_name(item.input_path)
                    )
                    if tokens and not item.sizes_text.strip():
                        item.sizes_text = "\n".join(tokens)
                    # Rank the library once. The UI reveals 20 more results per click,
                    # so expanding candidates never reruns the visual model.
                    item.candidates = self.index.search(
                        item.input_path, top_k=len(self.index.paths)
                    )
                    item.candidate_limit = 20
                    item.decision_stage = ""
                    item.output_unmatched = False
                    can_auto_confirm = (
                        self.engine.session is not None
                        and item.candidates
                        and item.candidates[0].score >= .82
                        and (
                            len(item.candidates) == 1
                            or item.candidates[0].score - item.candidates[1].score >= .04
                        )
                    )
                    item.selected = item.candidates[0].path if can_auto_confirm else None
                    item.decision_stage = "AI确认" if item.selected else ""
                    item.status = "AI已确认" if item.selected else "待人工确认"
                except Exception as exc:
                    item.status = f"失败：{exc}"
                finally:
                    item.analyzed = True
                self.events.put(("analysis_progress", i + 1))
            self.events.put(("analysis_done",))

        threading.Thread(target=worker, daemon=True).start()

    def _refresh_candidates(self, item: WorkItem, focus_index: int | None = None):
        self.candidate_list.delete(0, tk.END)
        visible = item.candidates[:item.candidate_limit]
        self.candidate_header_var.set(
            f"候选款式（双击确认）｜显示 {len(visible)}/{len(item.candidates)}"
            if item.candidates else "候选款式｜等待分析"
        )
        self.more_candidates_button.configure(
            state=(tk.NORMAL if item.candidate_limit < len(item.candidates) else tk.DISABLED)
        )
        for match in visible:
            self.candidate_list.insert(tk.END, f"{match.score * 100:5.1f}%  {match.path.name}")
        if visible:
            chosen_index = focus_index
            if chosen_index is None or not 0 <= chosen_index < len(visible):
                chosen_index = next(
                    (i for i, m in enumerate(visible) if m.path == item.selected), 0
                )
            self.candidate_list.selection_set(chosen_index)
            self.candidate_list.activate(chosen_index)
            self.candidate_list.see(chosen_index)
            self._show_image(visible[chosen_index].path, self.candidate_canvas, "candidate_photo")
        else:
            self._set_preview_view("candidate_photo", None)
            self.candidate_photo = None
            self.candidate_canvas.delete("all")

    def _show_more_candidates(self):
        if self.current_index is None:
            self.status_var.set("请先选择一张待处理照片")
            return
        item = self.items[self.current_index]
        if not item.candidates:
            self.status_var.set("请先分析当前照片")
            return
        if item.candidate_limit >= len(item.candidates):
            self.status_var.set(f"已显示图库全部 {len(item.candidates)} 款候选")
            return
        first_new_index = item.candidate_limit
        item.candidate_limit = min(len(item.candidates), item.candidate_limit + 20)
        self._refresh_candidates(item, focus_index=first_new_index)
        visible_count = max(item.candidate_limit, 1)
        self.candidate_list.yview_moveto(first_new_index / visible_count)
        if item.candidate_limit >= len(item.candidates):
            self.status_var.set(f"已显示图库全部 {len(item.candidates)} 款候选")
        else:
            self.status_var.set(
                f"候选款式已显示 {item.candidate_limit}/{len(item.candidates)} 款；可继续再显示20款"
            )

    def _on_candidate_select(self, _event=None):
        if self.current_index is None:
            return
        selection = self.candidate_list.curselection()
        item = self.items[self.current_index]
        if selection and selection[0] < len(item.candidates):
            self._show_image(item.candidates[selection[0]].path, self.candidate_canvas, "candidate_photo")

    def _confirm_candidate(self, _event=None):
        if self.current_index is None:
            return
        selection = self.candidate_list.curselection()
        item = self.items[self.current_index]
        if not selection or selection[0] >= len(item.candidates):
            return
        selected = item.candidates[selection[0]].path
        item.selected = selected
        item.output_unmatched = False
        item.decision_stage = "AI候选确认"
        item.status = item.decision_stage
        if self.index:
            self.index.record_feedback(item.input_path, selected)
        self._save_fields(show_preview=True)
        self._refresh_input_list()
        decided_index = self.current_index
        self._advance_after_decision(decided_index)
        return "break"

    def _manual_select(self):
        if self.current_index is None:
            return
        initial = self.library_var.get() or None
        path = filedialog.askopenfilename(
            title="选择正确款式图",
            initialdir=initial,
            filetypes=[("图片", "*.png *.jpg *.jpeg *.webp *.bmp")],
        )
        if path:
            item = self.items[self.current_index]
            selected = Path(path)
            item.selected = selected
            item.output_unmatched = False
            item.decision_stage = "人工确认"
            item.status = item.decision_stage
            if self.index:
                self.index.record_feedback(item.input_path, selected)
            self._save_fields(show_preview=True)
            self._refresh_input_list()
            self._advance_after_decision(self.current_index)

    def _mark_unmatched(self):
        if self.current_index is None:
            return
        item = self.items[self.current_index]
        item.selected = item.input_path
        item.output_unmatched = True
        item.decision_stage = "未匹配"
        item.status = "未匹配（输出实拍图）"
        self._save_fields(show_preview=True)
        self._refresh_input_list()
        self.status_var.set("✓ 已标记未匹配；右侧已显示带“未匹配”红字的预览")
        self._advance_after_decision(self.current_index)

    def _save_fields(self, show_preview: bool = False, silent: bool = False):
        if self.current_index is None or self.current_index >= len(self.items):
            return
        item = self.items[self.current_index]
        item.color = self._current_color()
        lines = []
        for size in self.size_order:
            value = self.size_vars[size].get().strip()
            if value:
                if value.isdigit():
                    lines.append(f"{size}{int(value)}")
                else:
                    lines.append(f"{size}{value}")
        extra = self.extra_textbox.get("1.0", "end-1c").strip()
        if extra:
            lines.append(extra)
        item.sizes_text = "\n".join(lines).strip()
        if show_preview:
            self._render_annotation_preview(item)
            self.status_var.set("✓ 已确认并保存；右侧已经显示最终红字效果预览")
        elif not silent:
            self.status_var.set("已保存：填写了数量的尺码和其他文字将在图片上标红")

    def _render_annotation_preview(self, item: WorkItem):
        source = item.selected
        if source is None:
            selection = self.candidate_list.curselection()
            if selection and selection[0] < len(item.candidates):
                source = item.candidates[selection[0]].path
            elif item.candidates:
                source = item.candidates[0].path
            else:
                source = item.input_path

        if item.output_unmatched:
            # The real photo already contains the handwritten size quantities.
            # Only add the unmatched state to avoid drawing duplicate red sizes.
            lines = ["未匹配"]
            preview_color = ""
        else:
            lines = annotation_lines(item.sizes_text)
            preview_color = item.color
        preview_dir = app_data_dir() / "previews"
        preview_path = preview_dir / f"preview_{self.current_index}.jpg"
        annotate_result(source, preview_path, preview_color, lines)
        self._show_image(preview_path, self.candidate_canvas, "candidate_photo")

    def _on_sizes_return(self, event):
        # Shift+Enter remains available for a new line; plain Enter confirms the field.
        if event.state & 0x0001:
            return None
        self._save_fields(show_preview=True)
        return "break"

    def _remove_current(self):
        if self.current_index is None:
            return
        removed_index = self.current_index
        del self.items[removed_index]
        self.current_index = None
        self._refresh_input_list()
        if self.items:
            self._select_item(min(removed_index, len(self.items) - 1))
        else:
            self.current_var.set("")
            self._set_preview_view("input_photo", None)
            self._set_preview_view("candidate_photo", None)
            self.input_photo = None
            self.candidate_photo = None
            self.input_canvas.delete("all")
            self.candidate_canvas.delete("all")

    def _export_all(self):
        self._save_fields()
        if not self.items:
            messagebox.showwarning("提示", "没有待导出的照片")
            return
        selected_items = [
            (position, item)
            for position, item in enumerate(self.items, start=1)
            if item.selected
        ]
        skipped = len(self.items) - len(selected_items)
        if skipped:
            proceed = messagebox.askokcancel(
                "还有照片未确认",
                f"当前还有 {skipped} 张照片尚未确认。\n\n"
                "点击“确定”：忽略这些未确认照片，继续输出已确认结果。\n"
                "点击“取消”：返回程序继续确认。",
            )
            if not proceed:
                self.status_var.set(f"已返回：还有 {skipped} 张照片需要确认")
                return
        if not selected_items:
            messagebox.showwarning("提示", "还没有选择任何款式图。AI找不到时请点“手动选择款式图”。")
            return
        export_items = selected_items

        root = Path(self.output_var.get() or (Path(self.library_var.get()) / "标注结果"))
        task_name = next_task_name(root)
        batch = root / task_name
        batch.mkdir(parents=True, exist_ok=True)
        detail_rows = []
        summary: dict[tuple[str, str, str], int] = {}
        overview_entries: list[tuple[Path, str]] = []

        for position, item in export_items:
            image_number = f"图片{position:02d}"
            color = item.color.strip() or "未识别颜色"
            selected = item.selected
            stage = item.decision_stage or "人工确认"
            # Keep the catalog image's original filename exactly as requested.
            destination = batch / selected.name
            raw_text = item.sizes_text.strip()
            lines = annotation_lines(raw_text)
            annotation_color = color
            if item.output_unmatched:
                lines = ["未匹配"]
                annotation_color = ""
            if not lines and not raw_text:
                lines = ["无尺码数量"]
            annotate_result(selected, destination, annotation_color, lines)
            detail_rows.append([
                task_name, image_number, stage, selected.name, color,
                raw_text if raw_text else "无尺码数量",
                str(destination.resolve()),
            ])
            overview_entries.append((destination, f"{image_number}  {stage}"))
            tokens = parse_size_tokens(raw_text)
            for token in tokens:
                match = re.match(r"(.+?)(\d+)$", token)
                if match:
                    size, quantity = match.groups()
                    key = (selected.name, color, size)
                    summary[key] = summary.get(key, 0) + int(quantity)
            if raw_text and not tokens:
                summary[(selected.name, color, raw_text)] = ""

        detail_path = batch / "任务明细.csv"
        with detail_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["任务编号", "图片编号", "确认阶段", "款式图片", "颜色", "尺码数量", "结果图绝对路径"])
            writer.writerows(detail_rows)

        summary_path = batch / "补货汇总.csv"
        with summary_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["款式图片", "颜色", "尺码", "合计数量"])
            for (style, color, size), quantity in sorted(summary.items()):
                writer.writerow([style, color, size, quantity])

        overview_path = batch / f"{task_name}_总览.jpg"
        create_overview(overview_entries, overview_path)
        self.status_var.set(f"{task_name} 已导出 {len(export_items)} 张结果")
        self._open_folder(batch)
        messagebox.showinfo(
            "完成",
            f"{task_name} 已完成\n已导出 {len(export_items)} 张"
            + (f"，另有 {skipped} 张未选择，未处理" if skipped else "")
            + f"\n结果保存到：\n{batch}",
        )

    @staticmethod
    def _open_folder(path: Path):
        try:
            os.startfile(path)
        except AttributeError:
            subprocess.Popen(["explorer", str(path)])

    def _poll_events(self):
        if self._closing:
            return
        try:
            while True:
                event = self.events.get_nowait()
                kind = event[0]
                if kind == "index_progress":
                    _, done, total, name = event
                    self.progress["maximum"] = max(total, 1)
                    self.progress["value"] = done
                    self.status_var.set(f"正在索引 {done}/{total}：{name}")
                elif kind == "index_done":
                    self.status_var.set(f"图库已就绪：{event[1]} 款｜{self.engine.mode}")
                    self._persist()
                    if getattr(self, "_index_rebuild_requested", False):
                        for item in self.items:
                            if not item.selected:
                                item.analyzed = False
                                item.candidates.clear()
                                item.status = "待分析"
                        self._index_rebuild_requested = False
                        self._refresh_input_list()
                elif kind == "analysis_progress":
                    self.progress["value"] = event[1]
                    self._refresh_input_list()
                elif kind == "analysis_done":
                    self.status_var.set("分析完成，请确认候选款式并修正颜色/尺码")
                    self._refresh_input_list()
                    if self.current_index is not None:
                        self._select_item(self.current_index)
                elif kind == "error":
                    messagebox.showerror("错误", event[1])
                    self.status_var.set("发生错误")
        except queue.Empty:
            pass
        if not self._closing:
            self._poll_after_id = self.after(100, self._poll_events)

    def destroy(self):
        """Cancel pending callbacks before Tk tears down the window."""
        self._closing = True
        for after_id in self._startup_after_ids:
            try:
                self.after_cancel(after_id)
            except tk.TclError:
                pass
        self._startup_after_ids.clear()
        if self._poll_after_id:
            try:
                self.after_cancel(self._poll_after_id)
            except tk.TclError:
                pass
        super().destroy()


if __name__ == "__main__":
    app = ClothingMatcherApp()
    app.mainloop()
