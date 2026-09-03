from hashlib import sha256
from pathlib import Path

from PIL import Image

from engine import (
    annotate_result,
    annotate_unconfirmed,
    annotation_lines,
    create_overview,
    detect_color_name,
    iter_images,
    next_task_name,
    parse_size_tokens,
)


ROOT = Path(__file__).resolve().parent.parent
LIBRARY = ROOT / "2026短袖"
TEMP = Path.home() / "AppData" / "Local" / "Temp"
OUTPUT = Path(__file__).resolve().parent / ".verification" / "22900_至尊_7_标注.png"
UNCONFIRMED_OUTPUT = Path(__file__).resolve().parent / ".verification" / "图片02_二次未确认.jpg"
OVERVIEW_OUTPUT = Path(__file__).resolve().parent / ".verification" / "任务测试_总览.jpg"
CUSTOM_OUTPUT = Path(__file__).resolve().parent / ".verification" / "custom_text_标注.png"


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> None:
    assert parse_size_tokens("M1 L2 XL1 2XL2") == ["M1", "L2", "XL1", "2XL2"]
    assert parse_size_tokens("xxl3, XXXL1") == ["2XL3", "3XL1"]
    assert annotation_lines("M1 XL1 2XL2") == ["M1", "XL1", "2XL2"]
    assert annotation_lines("成人款 3件") == ["成人款 3件"]
    source = LIBRARY / "22900_至尊_7.png"
    before = digest(source)
    annotate_result(source, OUTPUT, "灰白条纹", ["XL1", "2XL1"])
    annotate_result(source, CUSTOM_OUTPUT, "深灰色", annotation_lines("成人款 3件\n备用文字"))
    after = digest(source)
    assert before == after, "source image changed"
    assert OUTPUT.exists() and OUTPUT.stat().st_size > 10_000
    assert CUSTOM_OUTPUT.exists() and CUSTOM_OUTPUT.stat().st_size > 10_000
    with Image.open(OUTPUT) as image:
        assert image.width > 500 and image.height > 500
    names = [path.name for path in iter_images(LIBRARY)]
    assert not any("_标注" in name for name in names)
    input_path = TEMP / "codex-clipboard-9933f5de-1281-429f-9a42-081c8f861473.jpg"
    if input_path.exists():
        input_before = digest(input_path)
        annotate_unconfirmed(input_path, UNCONFIRMED_OUTPUT)
        assert input_before == digest(input_path), "input photo changed"
        create_overview(
            [(OUTPUT, "图片01  初检确认"), (UNCONFIRMED_OUTPUT, "图片02  未确认")],
            OVERVIEW_OUTPUT,
        )
        assert OVERVIEW_OUTPUT.exists()
    assert next_task_name(OUTPUT.parent).startswith("任务")
    print(f"library_images={len(names)}")
    print(f"source_unchanged={before == after}")
    print(f"output={OUTPUT}")
    for filename in (
        "codex-clipboard-cb0fc386-fa1a-41a4-bc63-0ae47c9d024f.jpg",
        "codex-clipboard-9933f5de-1281-429f-9a42-081c8f861473.jpg",
        "codex-clipboard-bab8273b-a9a9-42c4-b5b4-98e4bb35594c.jpg",
    ):
        path = TEMP / filename
        if path.exists():
            print(f"color {filename} => {detect_color_name(path)}")


if __name__ == "__main__":
    main()
