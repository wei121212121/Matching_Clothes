from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np


# Keep verification artifacts away from the installed app and V7.
_test_data = Path(__file__).resolve().parent / ".verification" / "v8_testdata"
_test_data.mkdir(parents=True, exist_ok=True)
os.environ["LOCALAPPDATA"] = str(_test_data)

from app import ClothingMatcherApp, WorkItem  # noqa: E402
from engine import (  # noqa: E402
    LibraryIndex,
    Match,
    app_data_dir,
    detect_color_name,
    detect_red_text_tokens,
    parse_marked_color_text,
)


ROOT = Path(__file__).resolve().parent.parent
LIBRARY = ROOT / "2026短袖 测试"
TEMP = Path.home() / "AppData" / "Local" / "Temp"


class FakeEngine:
    mode = "V8 feedback verification"
    session = object()

    def __init__(self, vectors: dict[str, np.ndarray]):
        self.vectors = vectors

    def embed(self, path: Path) -> np.ndarray:
        return self.vectors[path.name]


def test_small_screen_layout() -> None:
    sample = LIBRARY / "22900_至尊_21.png"
    window = ClothingMatcherApp()
    window.geometry("700x480")
    window.update()
    window.items = [
        WorkItem(
            sample,
            candidates=[Match(sample, .5)],
            analyzed=True,
            status="待人工确认",
        )
    ]
    window._refresh_input_list()
    window._select_item(0)
    window.update()
    bottom = window.winfo_rooty() + window.winfo_height()
    assert window.export_button.winfo_ismapped()
    assert window.export_button.winfo_rooty() + window.export_button.winfo_height() <= bottom
    assert window.candidate_actions.winfo_ismapped()
    window.destroy()


def test_default_pane_sizes() -> None:
    window = ClothingMatcherApp()
    window.geometry("1400x800")
    window.update()
    window._set_default_pane_sizes()
    window.update()
    width = window.main_panes.winfo_width()
    positions = [window.main_panes.sash_coord(i)[0] for i in range(3)]
    assert positions[0] < width * 0.23, (width, positions)
    assert width * 0.40 < positions[1] < width * 0.53, (width, positions)
    assert width * 0.68 < positions[2] < width * 0.82, (width, positions)
    assert window.auto_next_check.winfo_ismapped()
    # Wrapped editor: preset colors and all five size inputs are laid out over
    # multiple rows instead of extending beyond the right edge.
    assert window.color_other_frame.grid_info()["row"] == 1
    assert {child.grid_info().get("row") for child in window.size_frame.winfo_children()} == {0, 1}
    window.destroy()


def test_feedback_learning() -> None:
    query_a = TEMP / "codex-clipboard-db4500ae-62dc-4920-9f64-48324c543c68.jpg"
    query_b = TEMP / "codex-clipboard-cb0fc386-fa1a-41a4-bc63-0ae47c9d024f.jpg"
    if not query_a.exists() or not query_b.exists():
        return
    wrong = LIBRARY / "19900_名洋_4.png"
    correct = LIBRARY / "22900_至尊_19.png"
    engine = FakeEngine(
        {
            query_a.name: np.asarray([1.0, 0.0], dtype=np.float32),
            query_b.name: np.asarray([0.95, 0.31], dtype=np.float32),
        }
    )
    index = LibraryIndex(LIBRARY, engine)
    index.paths = [wrong, correct]
    index.vectors = np.asarray([[0.60, 0.80], [0.40, 0.916]], dtype=np.float32)
    index.texts = ["", ""]
    index.feedback_path.unlink(missing_ok=True)
    index.search(query_a, top_k=2)
    index.record_feedback(query_a, correct)
    learned = index.search(query_b, top_k=2)
    assert learned[0].path == correct


def test_preview_zoom_and_pan() -> None:
    sample = LIBRARY / "22900_至尊_21.png"
    window = ClothingMatcherApp()
    window.geometry("900x650")
    window.update()
    window._show_image(sample, window.input_canvas, "input_photo")
    window.update()
    assert window.input_view is not None
    window._on_preview_zoom(
        SimpleNamespace(x=120, y=120, delta=120), "input_photo"
    )
    assert window.input_view["zoom"] > 1.0
    window._start_preview_pan(SimpleNamespace(x=120, y=120), "input_photo")
    window._move_preview_pan(SimpleNamespace(x=150, y=155), "input_photo")
    assert window.input_view["offset_x"] != 0 or window.input_view["offset_y"] != 0
    window._reset_preview("input_photo")
    assert window.input_view["zoom"] == 1.0
    assert window.input_view["offset_x"] == 0.0
    assert window.input_view["offset_y"] == 0.0
    window.destroy()


def test_color_examples() -> None:
    expected = {
        "codex-clipboard-cb0fc386-fa1a-41a4-bc63-0ae47c9d024f.jpg": "白色",
        "codex-clipboard-9933f5de-1281-429f-9a42-081c8f861473.jpg": "灰色",
        "codex-clipboard-bab8273b-a9a9-42c4-b5b4-98e4bb35594c.jpg": "黑色",
    }
    for filename, color in expected.items():
        path = TEMP / filename
        if path.exists():
            assert detect_color_name(path) == color, (filename, detect_color_name(path), color)


def test_marked_color_priority() -> None:
    assert parse_marked_color_text("M2 XL1 浅灰") == "浅灰色"
    assert parse_marked_color_text("L1 绿") == "绿色"
    marked = TEMP / "codex-clipboard-5347fd38-f6d3-4e6c-84d5-0b6d13b9a856.jpg"
    if marked.exists():
        tokens, text = detect_red_text_tokens(marked)
        assert tokens == ["M1", "L1", "XL1"], (tokens, text)
        assert parse_marked_color_text(text) == "绿色", text


if __name__ == "__main__":
    assert app_data_dir().name == "ClothingMatcherV8"
    test_small_screen_layout()
    test_default_pane_sizes()
    test_feedback_learning()
    test_preview_zoom_and_pan()
    test_color_examples()
    test_marked_color_priority()
    print("v8_small_screen=ok")
    print("v8_default_panes=ok")
    print("v8_feedback_learning=ok")
    print("v8_preview_zoom_pan=ok")
    print("v8_color_examples=ok")
    print("v8_marked_color_priority=ok")
