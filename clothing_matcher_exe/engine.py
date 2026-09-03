from __future__ import annotations

import hashlib
import json
import os
import re
from difflib import SequenceMatcher
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
SIZE_TOKEN_RE = re.compile(r"(?<![A-Z0-9])((?:[2-9]XL|XXXL|XXL|XL|L|M|S))\s*([0-9]+)", re.I)


@dataclass
class Match:
    path: Path
    score: float


def app_data_dir() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA", Path.home()))
    path = base / "ClothingMatcher"
    path.mkdir(parents=True, exist_ok=True)
    return path


def iter_images(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.rglob("*")
        if (
            p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
            and "_标注" not in p.stem
            and not any("标注" in part or part == "找款结果" for part in p.parts)
        )
    )


def _crop_garment(im: Image.Image) -> Image.Image:
    """Crop likely garment area while retaining the chest print."""
    w, h = im.size
    if h > w:
        box = (int(w * 0.06), int(h * 0.08), int(w * 0.94), int(h * 0.90))
    else:
        box = (int(w * 0.08), int(h * 0.06), int(w * 0.92), int(h * 0.94))
    return im.crop(box)


def _neutralize_red_annotations(im: Image.Image) -> Image.Image:
    """Replace bright-red stock labels with nearby blurred pixels for matching."""
    preview = im.copy()
    arr = np.asarray(preview, dtype=np.uint8)
    red = (
        (arr[:, :, 0] > 165)
        & (arr[:, :, 0].astype(np.float32) > arr[:, :, 1] * 1.35)
        & (arr[:, :, 0].astype(np.float32) > arr[:, :, 2] * 1.25)
    )
    if not red.any():
        return preview
    mask = Image.fromarray(np.uint8(red) * 255).filter(ImageFilter.MaxFilter(11))
    replacement = preview.filter(ImageFilter.GaussianBlur(radius=14))
    preview.paste(replacement, mask=mask)
    return preview


def _classic_descriptor(path: Path) -> np.ndarray:
    """Color-light descriptor used when the optional ONNX model is absent."""
    with Image.open(path) as src:
        im = _crop_garment(ImageOps.exif_transpose(src).convert("RGB"))
        im.thumbnail((256, 256), Image.Resampling.LANCZOS)
        im = _neutralize_red_annotations(im)
        gray = np.asarray(im.resize((48, 48), Image.Resampling.BILINEAR).convert("L"), dtype=np.float32)
        gray = (gray - gray.mean()) / (gray.std() + 1e-5)

        gx = np.zeros_like(gray)
        gy = np.zeros_like(gray)
        gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
        gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
        magnitude = np.sqrt(gx * gx + gy * gy)
        angle = (np.arctan2(gy, gx) + np.pi) % np.pi

        blocks: list[np.ndarray] = []
        for by in range(4):
            for bx in range(4):
                ys = slice(by * 12, (by + 1) * 12)
                xs = slice(bx * 12, (bx + 1) * 12)
                hist, _ = np.histogram(
                    angle[ys, xs],
                    bins=8,
                    range=(0, np.pi),
                    weights=magnitude[ys, xs],
                )
                blocks.append(hist.astype(np.float32))

        edge_small = np.asarray(
            Image.fromarray(np.uint8(np.clip(magnitude / (magnitude.max() + 1e-5) * 255, 0, 255)))
            .resize((24, 24), Image.Resampling.BILINEAR),
            dtype=np.float32,
        ).ravel()

        # Low-weight color histogram helps separate grossly different items without
        # dominating same-style/different-color matches.
        rgb = np.asarray(im.resize((64, 64), Image.Resampling.BILINEAR), dtype=np.uint8)
        color_hist = []
        for channel in range(3):
            hist, _ = np.histogram(rgb[:, :, channel], bins=12, range=(0, 256), density=True)
            color_hist.extend(hist * 0.15)

        desc = np.concatenate([*blocks, edge_small, np.asarray(color_hist, dtype=np.float32)])
        norm = np.linalg.norm(desc)
        return desc / (norm + 1e-8)


class EmbeddingEngine:
    """CPU visual embedding engine with a deterministic lightweight fallback."""

    def __init__(self, model_path: Path | None = None):
        self.model_path = model_path
        self.session = None
        self.model_kind = "classic"
        self.mode = "轻量视觉特征"
        if model_path and model_path.exists():
            try:
                import onnxruntime as ort

                options = ort.SessionOptions()
                options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                options.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
                self.session = ort.InferenceSession(
                    str(model_path),
                    sess_options=options,
                    providers=["CPUExecutionProvider"],
                )
                if "sscd" in model_path.name.lower():
                    self.model_kind = "sscd"
                    self.mode = "SSCD 双视角 + 纹理 + OCR文字融合（CPU）"
                elif "dinov2" in model_path.name.lower():
                    self.model_kind = "dinov2"
                    self.mode = "DINOv2 Q4 + 纹理组合（CPU）"
                else:
                    self.model_kind = "clip"
                    self.mode = "CLIP ONNX"
            except Exception:
                self.session = None

    def embed(self, path: Path) -> np.ndarray:
        if self.session is None:
            return _classic_descriptor(path)

        if self.model_kind == "sscd":
            return self._embed_sscd(path)

        if self.model_kind == "dinov2":
            return self._embed_dinov2(path)

        with Image.open(path) as src:
            im = _crop_garment(ImageOps.exif_transpose(src).convert("RGB"))
            w, h = im.size
            scale = 224 / min(w, h)
            resized = im.resize((round(w * scale), round(h * scale)), Image.Resampling.BICUBIC)
            left = max(0, (resized.width - 224) // 2)
            top = max(0, (resized.height - 224) // 2)
            resized = resized.crop((left, top, left + 224, top + 224))
            resized = _neutralize_red_annotations(resized)
            arr = np.asarray(resized, dtype=np.float32) / 255.0
            mean = np.asarray([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
            std = np.asarray([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
            arr = ((arr - mean) / std).transpose(2, 0, 1)[None, :]

        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: arr})
        # Xenova vision-only CLIP exports projected image embeddings as one output.
        candidates = [np.asarray(x) for x in outputs if np.asarray(x).ndim == 2]
        vector = candidates[-1][0].astype(np.float32)
        return vector / (np.linalg.norm(vector) + 1e-8)

    def _embed_sscd(self, path: Path) -> np.ndarray:
        with Image.open(path) as src:
            base = _crop_garment(ImageOps.exif_transpose(src).convert("RGB"))
        w, h = base.size
        print_crop = base.crop((int(.08 * w), int(.16 * h), int(.92 * w), int(.82 * h)))
        prepared = []
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        for im in (base, print_crop):
            # SSCD explicitly recommends square resize for copy detection.
            im = im.resize((320, 320), Image.Resampling.BICUBIC)
            im = _neutralize_red_annotations(im)
            im = Image.blend(im, im.convert("L").convert("RGB"), 0.20)
            arr = np.asarray(im, dtype=np.float32) / 255.0
            prepared.append(((arr - mean) / std).transpose(2, 0, 1))
        input_name = self.session.get_inputs()[0].name
        learned = np.asarray(
            self.session.run(None, {input_name: np.stack(prepared)})[0],
            dtype=np.float32,
        )
        learned /= np.linalg.norm(learned, axis=1, keepdims=True) + 1e-8
        texture = _classic_descriptor(path)
        combined = np.concatenate(
            [
                learned[0] * np.sqrt(0.55),
                learned[1] * np.sqrt(0.35),
                texture * np.sqrt(0.10),
            ]
        ).astype(np.float32)
        return combined / (np.linalg.norm(combined) + 1e-8)

    def _embed_dinov2(self, path: Path) -> np.ndarray:
        """DINOv2 structure embedding blended with color-light local texture."""
        arr = self._prepare_dinov2(path)
        input_name = self.session.get_inputs()[0].name
        hidden = np.asarray(self.session.run(None, {input_name: arr})[0])[0]
        return self._combine_dinov2(hidden, path)

    def _prepare_dinov2(self, path: Path) -> np.ndarray:
        with Image.open(path) as src:
            im = _crop_garment(ImageOps.exif_transpose(src).convert("RGB"))
            im = _neutralize_red_annotations(im)
            # Suppress part of the color signal so that the same style in another
            # color remains close, while retaining enough shading for fabric detail.
            gray_rgb = im.convert("L").convert("RGB")
            im = Image.blend(im, gray_rgb, 0.35)
            scale = 256 / min(im.size)
            im = im.resize(
                (round(im.width * scale), round(im.height * scale)),
                Image.Resampling.BICUBIC,
            )
            left = max(0, (im.width - 224) // 2)
            top = max(0, (im.height - 224) // 2)
            im = im.crop((left, top, left + 224, top + 224))
            arr = np.asarray(im, dtype=np.float32) / 255.0
            mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
            arr = ((arr - mean) / std).transpose(2, 0, 1)[None, :]
        return arr

    def _combine_dinov2(self, hidden: np.ndarray, path: Path) -> np.ndarray:
        cls_vector = hidden[0].astype(np.float32)
        patch_vector = hidden[1:].mean(axis=0).astype(np.float32)
        learned = cls_vector * 0.65 + patch_vector * 0.35
        learned /= np.linalg.norm(learned) + 1e-8

        texture = _classic_descriptor(path)
        combined = np.concatenate(
            [learned * np.sqrt(0.82), texture * np.sqrt(0.18)]
        ).astype(np.float32)
        return combined / (np.linalg.norm(combined) + 1e-8)

    def embed_many(self, paths: Sequence[Path], batch_size: int = 8) -> list[np.ndarray]:
        if self.session is None or self.model_kind not in {"dinov2"}:
            return [self.embed(path) for path in paths]
        vectors: list[np.ndarray] = []
        input_name = self.session.get_inputs()[0].name
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start:start + batch_size]
            arrays = np.concatenate([self._prepare_dinov2(path) for path in batch_paths], axis=0)
            hidden_batch = np.asarray(self.session.run(None, {input_name: arrays})[0])
            vectors.extend(
                self._combine_dinov2(hidden, path)
                for hidden, path in zip(hidden_batch, batch_paths)
            )
        return vectors


class LibraryIndex:
    def __init__(self, library: Path, engine: EmbeddingEngine):
        self.library = library.resolve()
        self.engine = engine
        self.paths: list[Path] = []
        self.vectors: np.ndarray | None = None
        self.texts: list[str] = []

    @property
    def cache_path(self) -> Path:
        key = hashlib.sha1(str(self.library).encode("utf-8")).hexdigest()[:16]
        return app_data_dir() / f"index-{key}.npz"

    def build(self, progress: Callable[[int, int, str], None] | None = None) -> int:
        paths = iter_images(self.library)
        vectors: list[np.ndarray] = []
        texts: list[str] = []
        for start in range(0, len(paths), 8):
            batch_paths = paths[start:start + 8]
            try:
                batch_vectors = self.engine.embed_many(batch_paths, batch_size=8)
            except Exception:
                batch_vectors = []
                for path in batch_paths:
                    try:
                        batch_vectors.append(self.engine.embed(path))
                    except Exception:
                        batch_vectors.append(np.zeros(1, dtype=np.float32))
            vectors.extend(batch_vectors)
            if progress:
                for offset, path in enumerate(batch_paths, start=1):
                    progress(start + offset, len(paths), path.name)
            texts.extend(detect_design_text(path) for path in batch_paths)

        valid_dim = max((v.size for v in vectors), default=0)
        normalized = [
            v if v.size == valid_dim else np.zeros(valid_dim, dtype=np.float32)
            for v in vectors
        ]
        self.paths = paths
        self.vectors = np.vstack(normalized) if normalized else np.empty((0, 0), dtype=np.float32)
        self.texts = texts
        np.savez_compressed(
            self.cache_path,
            paths=np.asarray([str(p) for p in paths]),
            mtimes=np.asarray([p.stat().st_mtime_ns for p in paths], dtype=np.int64),
            vectors=self.vectors,
            texts=np.asarray(self.texts, dtype=np.str_),
            mode=np.asarray([self.engine.mode]),
        )
        return len(paths)

    def load(self) -> bool:
        if not self.cache_path.exists():
            return False
        try:
            data = np.load(self.cache_path, allow_pickle=False)
            paths = [Path(x) for x in data["paths"].tolist()]
            mtimes = data["mtimes"].tolist()
            if len(paths) != len(mtimes):
                return False
            if paths != iter_images(self.library):
                return False
            if any(not p.exists() or p.stat().st_mtime_ns != m for p, m in zip(paths, mtimes)):
                return False
            if data["mode"].tolist()[0] != self.engine.mode:
                return False
            if "texts" not in data or len(data["texts"]) != len(paths):
                return False
            self.paths = paths
            self.vectors = data["vectors"].astype(np.float32)
            self.texts = data["texts"].tolist()
            return True
        except Exception:
            return False

    def search(self, query_path: Path, top_k: int = 5) -> list[Match]:
        if self.vectors is None or not len(self.paths):
            raise RuntimeError("图库尚未建立索引")
        query = self.engine.embed(query_path)
        if query.size != self.vectors.shape[1]:
            raise RuntimeError("索引模型不一致，请重新建立索引")
        scores = self.vectors @ query
        query_text = detect_design_text(query_path)
        if query_text and self.texts:
            text_scores = np.asarray(
                [design_text_similarity(query_text, text) for text in self.texts],
                dtype=np.float32,
            )
            if float(text_scores.max(initial=0.0)) >= 0.72:
                scores = scores * 0.65 + text_scores * 0.35
        order = np.argsort(scores)[::-1][:top_k]
        return [Match(self.paths[i], float(scores[i])) for i in order]


def normalize_size_token(size: str, quantity: str) -> str:
    size = size.upper()
    size = {"XXL": "2XL", "XXXL": "3XL"}.get(size, size)
    return f"{size}{int(quantity)}"


def parse_size_tokens(text: str) -> list[str]:
    normalized = text.upper()
    tokens = [normalize_size_token(a, b) for a, b in SIZE_TOKEN_RE.findall(normalized)]
    seen = set()
    return [x for x in tokens if not (x in seen or seen.add(x))]


def annotation_lines(text: str) -> list[str]:
    """Return user-entered lines for red annotation without requiring size syntax."""
    raw = text.replace("\r\n", "\n").replace("\r", "\n")
    lines: list[str] = []
    for line in re.split(r"[\n,，、;；]+", raw):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        parsed_parts = [parse_size_tokens(part) for part in parts]
        if parts and len(parts) > 1 and all(len(parsed) == 1 for parsed in parsed_parts):
            lines.extend(parsed[0] for parsed in parsed_parts)
        else:
            lines.append(line)
    return lines


_OCR_ENGINE = None
_FULL_OCR_CACHE: dict[tuple[str, int], list[str]] = {}


def _full_ocr_lines(path: Path) -> list[str]:
    try:
        key = (str(path.resolve()), path.stat().st_mtime_ns)
        cached = _FULL_OCR_CACHE.get(key)
        if cached is not None:
            return cached
        from rapidocr_onnxruntime import RapidOCR

        global _OCR_ENGINE
        if _OCR_ENGINE is None:
            _OCR_ENGINE = RapidOCR()
        with Image.open(path) as src:
            image = ImageOps.exif_transpose(src).convert("RGB")
            image.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
        result, _ = _OCR_ENGINE(np.asarray(image))
        lines = [str(row[1]) for row in (result or [])]
        if len(_FULL_OCR_CACHE) > 512:
            _FULL_OCR_CACHE.clear()
        _FULL_OCR_CACHE[key] = lines
        return lines
    except Exception:
        return []


def detect_design_text(path: Path) -> str:
    """Extract stable Latin design words while ignoring stock codes and sizes."""
    tokens: list[str] = []
    for line in _full_ocr_lines(path):
        for token in re.findall(r"[A-Z0-9]+", line.upper()):
            if token.isdigit() or len(token) < 4:
                continue
            if SIZE_TOKEN_RE.fullmatch(token):
                continue
            if token not in tokens:
                tokens.append(token)
    return " ".join(tokens)


def design_text_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    a = left.split()
    b = right.split()
    best = 0.0
    for one in a:
        for two in b:
            if one == two and len(one) >= 6:
                return 1.0
            best = max(best, SequenceMatcher(None, one, two).ratio())
    return best


def detect_red_text_tokens(path: Path) -> tuple[list[str], str]:
    """Read red stock labels with RapidOCR; fall back to manual entry."""
    try:
        from rapidocr_onnxruntime import RapidOCR

        global _OCR_ENGINE
        if _OCR_ENGINE is None:
            _OCR_ENGINE = RapidOCR()
        with Image.open(path) as src:
            image = ImageOps.exif_transpose(src).convert("RGB")
            image.thumbnail((1400, 1400), Image.Resampling.LANCZOS)
        arr = np.asarray(image, dtype=np.uint8)
        red = (
            (arr[:, :, 0] > 155)
            & (arr[:, :, 0].astype(np.float32) > arr[:, :, 1] * 1.28)
            & (arr[:, :, 0].astype(np.float32) > arr[:, :, 2] * 1.18)
        )
        mask = Image.fromarray(np.uint8(red) * 255).filter(ImageFilter.MaxFilter(3))
        isolated = np.full((*red.shape, 3), 255, dtype=np.uint8)
        isolated[np.asarray(mask) > 0] = 0
        result, _ = _OCR_ENGINE(isolated)
        text = "\n".join(row[1] for row in (result or []))
        tokens = parse_size_tokens(text)
        if not tokens:
            result, _ = _OCR_ENGINE(np.asarray(image))
            text = "\n".join(row[1] for row in (result or []))
            tokens = parse_size_tokens(text)
        return tokens, text
    except Exception:
        return [], ""


def detect_color_name(path: Path) -> str:
    """Estimate garment color from center regions while ignoring bright red labels."""
    with Image.open(path) as src:
        im = ImageOps.exif_transpose(src).convert("RGB")
        w, h = im.size
        crop = np.asarray(im.crop((int(w * .20), int(h * .12), int(w * .80), int(h * .72))).resize((120, 120)))

    pixels = crop.reshape(-1, 3).astype(np.float32)
    red_label = (pixels[:, 0] > 180) & (pixels[:, 0] > pixels[:, 1] * 1.45) & (pixels[:, 0] > pixels[:, 2] * 1.35)
    near_white_bg = (pixels.min(axis=1) > 245)
    pixels = pixels[~red_label & ~near_white_bg]
    if len(pixels) == 0:
        return "未识别"

    # Favor the most common garment-like luminance cluster rather than graphics.
    lum = pixels.mean(axis=1)
    bins = np.clip((lum // 32).astype(int), 0, 7)
    common_bin = np.bincount(bins, minlength=8).argmax()
    chosen = pixels[bins == common_bin]
    rgb = np.median(chosen, axis=0)
    r, g, b = rgb / 255.0
    maxc, minc = max(r, g, b), min(r, g, b)
    saturation = (maxc - minc) / max(maxc, 1e-6)
    value = maxc

    if saturation < 0.13:
        # Dark fabric is often lifted by shop lighting and used to be reported as
        # gray. Treat the old dark-gray band as black and give black more headroom.
        if value < 0.55:
            return "黑色"
        if value < 0.82:
            return "灰色"
        return "白色"

    if r > g * 1.25 and r > b * 1.25:
        return "红色"
    if r > b * 1.35 and g > b * 1.15:
        return "卡其"
    if b > r * 1.20:
        return "蓝色"
    if g > r * 1.18:
        return "绿色"
    return "彩色"


def load_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "msyh.ttc",
        Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "msyhbd.ttc",
        Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "arialbd.ttf",
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def annotate_result(
    source: Path,
    destination: Path,
    color: str,
    size_tokens: Iterable[str],
) -> None:
    with Image.open(source) as src:
        im = ImageOps.exif_transpose(src).convert("RGBA")
    draw = ImageDraw.Draw(im)
    labels = [color.strip(), *[x.strip().upper() for x in size_tokens if x.strip()]]
    labels = [x for x in labels if x]
    font_size = max(38, min(96, im.width // 12))
    font = load_font(font_size)
    gap = int(font_size * 1.15)
    total_h = gap * len(labels)
    widths = [draw.textbbox((0, 0), label, font=font)[2] for label in labels]
    block_w = min(max(widths, default=font_size), int(im.width * .9))
    # A fixed center position is predictable across every style and color.
    x = max(0, (im.width - block_w) // 2)
    y = max(0, (im.height - total_h) // 2)
    for label in labels:
        box = draw.textbbox((0, 0), label, font=font)
        text_w = box[2] - box[0]
        text_x = x + max(0, (block_w - text_w) // 2)
        draw.text(
            (text_x, y),
            label,
            font=font,
            fill=(245, 45, 45, 255),
            stroke_width=0,
        )
        y += gap
    destination.parent.mkdir(parents=True, exist_ok=True)
    im.convert("RGB").save(destination, quality=95)


def _least_detailed_position(im: Image.Image, block_w: int, block_h: int) -> tuple[int, int]:
    """Choose a low-edge placement so labels are less likely to cover the main print."""
    w, h = im.size
    block_w = min(block_w, int(w * .9))
    block_h = min(block_h, int(h * .42))
    xs = [int(w * .05), max(0, (w - block_w) // 2), max(0, int(w * .95) - block_w)]
    ys = [int(h * .30), int(h * .50), max(0, int(h * .72) - block_h)]
    gray = np.asarray(im.resize((240, max(240, round(240 * h / w))), Image.Resampling.BILINEAR).convert("L"), dtype=np.float32)
    scale_x = gray.shape[1] / w
    scale_y = gray.shape[0] / h
    best = (xs[0], ys[0])
    best_score = float("inf")
    for x in xs:
        for y in ys:
            x1, x2 = int(x * scale_x), max(int((x + block_w) * scale_x), int(x * scale_x) + 1)
            y1, y2 = int(y * scale_y), max(int((y + block_h) * scale_y), int(y * scale_y) + 1)
            patch = gray[y1:y2, x1:x2]
            if patch.size == 0:
                continue
            edge = np.abs(np.diff(patch, axis=0)).mean() + np.abs(np.diff(patch, axis=1)).mean()
            score = float(edge + patch.std() * .18)
            if score < best_score:
                best_score = score
                best = (x, y)
    return best


def annotate_unconfirmed(source: Path, destination: Path) -> None:
    with Image.open(source) as src:
        im = ImageOps.exif_transpose(src).convert("RGB")
    draw = ImageDraw.Draw(im)
    font_size = max(56, min(150, im.width // 7))
    font = load_font(font_size)
    label = "未确认"
    box = draw.textbbox((0, 0), label, font=font)
    x = (im.width - (box[2] - box[0])) // 2
    y = int(im.height * .62)
    draw.text(
        (x, y),
        label,
        font=font,
        fill=(245, 45, 45),
        stroke_width=max(2, font_size // 35),
        stroke_fill=(140, 0, 0),
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    im.save(destination, quality=94)


def create_overview(
    entries: Sequence[tuple[Path, str]],
    destination: Path,
    columns: int = 3,
) -> None:
    if not entries:
        return
    tile_w, image_h, caption_h = 420, 420, 82
    rows = (len(entries) + columns - 1) // columns
    canvas = Image.new("RGB", (tile_w * columns, (image_h + caption_h) * rows), "white")
    font = load_font(30)
    draw = ImageDraw.Draw(canvas)
    for index, (path, caption) in enumerate(entries):
        row, column = divmod(index, columns)
        left, top = column * tile_w, row * (image_h + caption_h)
        with Image.open(path) as src:
            image = ImageOps.exif_transpose(src).convert("RGB")
            image.thumbnail((tile_w - 16, image_h - 16), Image.Resampling.LANCZOS)
        x = left + (tile_w - image.width) // 2
        y = top + (image_h - image.height) // 2
        canvas.paste(image, (x, y))
        draw.text((left + 12, top + image_h + 8), caption, font=font, fill=(30, 30, 30))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, quality=92)


def next_task_name(result_root: Path) -> str:
    highest = 0
    if result_root.exists():
        for child in result_root.iterdir():
            if child.is_dir():
                match = re.fullmatch(r"任务(\d{3,})", child.name)
                if match:
                    highest = max(highest, int(match.group(1)))
    return f"任务{highest + 1:03d}"


def save_settings(settings: dict) -> None:
    path = app_data_dir() / "settings.json"
    path.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")


def load_settings() -> dict:
    path = app_data_dir() / "settings.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
