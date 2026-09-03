import os
from pathlib import Path
from types import MethodType, SimpleNamespace


# Keep verification state inside the V8 workspace and away from installed/V7 data.
_test_data = Path(__file__).resolve().parent / ".verification" / "v8_import_testdata"
_test_data.mkdir(parents=True, exist_ok=True)
os.environ["LOCALAPPDATA"] = str(_test_data)

import app as app_module
from app import ClothingMatcherApp


ROOT = Path(__file__).resolve().parent.parent
LIBRARY = ROOT / "2026短袖 测试"
SAMPLE = LIBRARY / "19900_大创_1.png"


class DummyStatus:
    def __init__(self):
        self.value = ""

    def set(self, value):
        self.value = value


def test_folder_and_clipboard_import() -> None:
    fake = ClothingMatcherApp.__new__(ClothingMatcherApp)
    fake.items = []
    fake.current_index = 0
    fake.status_var = DummyStatus()
    fake._refresh_input_list = lambda: None
    fake._select_item = lambda _index: None
    fake._add_input_paths = MethodType(ClothingMatcherApp._add_input_paths, fake)
    fake._add_input_paths([LIBRARY])
    folder_count = len(fake.items)
    assert folder_count >= 160, folder_count

    original_grab = app_module.ImageGrab.grabclipboard
    try:
        app_module.ImageGrab.grabclipboard = lambda: [str(SAMPLE)]
        fake.items = []
        ClothingMatcherApp._paste_inputs(fake)
        assert len(fake.items) == 1 and fake.items[0].input_path == SAMPLE
    finally:
        app_module.ImageGrab.grabclipboard = original_grab
    print(f"folder_import={folder_count}")
    print("clipboard_file_import=ok")


def test_safe_drag_drop() -> None:
    window = ClothingMatcherApp()
    window.withdraw()
    window.items = []
    window.current_index = None
    event = SimpleNamespace(data=window.tk.call("list", str(SAMPLE)))
    window._on_safe_file_drop(event)
    window.update()
    assert len(window.items) == 1 and window.items[0].input_path == SAMPLE
    window.destroy()
    print("safe_ole_drag_drop=ok")


if __name__ == "__main__":
    test_folder_and_clipboard_import()
    test_safe_drag_drop()
