# Matching Clothes | Clothing Search and Restock Annotation Tool

[简体中文](README.md) | [English](README.en.md) | [한국어](README.ko.md)

Matching Clothes is a local Windows application that searches a changing style library for garments matching store photos. It reads red color, size, and quantity notes, lets a person confirm the correct result, and exports annotated images, task details, and restock summaries.

The application runs locally by default. Garment libraries, store photos, and results are not uploaded. AI ranks candidates, while the final match always remains a human decision.

## Features

- Four-view visual matching: full garment, center graphic, upper garment, and small chest logo.
- OCR text and lightweight texture fusion, with color treated only as weak evidence for matching the same style in another color.
- Red handwritten or overlaid color, size, and quantity notes are read before automatic color estimation.
- Shows 20 candidates by default, with progressive expansion and manual selection.
- Both preview panels support wheel zoom, left-button panning, and double-click reset.
- Confirmed choices can become local feedback samples for the current library only.
- Exports annotated images, a task overview, `任务明细.csv`, and `补货汇总.csv`.
- Source library images and store photos are never modified.

## Quick Start

### Requirements

- Windows 10 or 11
- Python 3.10 or later
- 8 GB RAM or more recommended

```powershell
git clone https://github.com/wei121212121/Matching_Clothes.git
cd Matching_Clothes\clothing_matcher_v8_ui_alt
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

The application can start without an ONNX model by using its built-in lightweight visual descriptor, but candidate accuracy will be lower. For the optimized pipeline, place a legally obtained `sscd_disc_mixup.onnx` file in `clothing_matcher_v8_ui_alt/models/`. Model weights are not included because of their size and separate licensing requirements.

### First Run

1. Select the style library and output directory.
2. Click **Build/Update Index**.
3. Import store photos, drag them into the application, or copy local files and press `Ctrl+V` in the left panel.
4. Click **Analyze New Photos (F5)**.
5. Compare the store photo and candidate preview, then verify the color and size quantities.
6. Confirm the correct candidate or choose **Unmatched** when the library has no corresponding style.
7. Click **Export All Results (Ctrl+E)** after all photos have been reviewed.

## Keyboard and Preview Controls

| Control | Action |
| --- | --- |
| `F5` | Analyze new photos |
| `Ctrl+Enter` | Confirm the current candidate |
| `Ctrl+Right` | Go to the next pending photo |
| `Ctrl+Left` | Go to the previous photo |
| `Ctrl+E` | Export the current batch |
| Mouse wheel | Zoom the active preview |
| Left-button drag | Pan a zoomed preview |
| Double-click | Fit the image to the preview |

## Workflow

Color alone must not determine a match. Compare distinctive graphics, stable text, print position, neckline, placket, pockets, stripe direction, fabric texture, and silhouette. Confirm at least two distinctive shared features, or a unique graphic/text feature together with a matching structural feature.

The first 20 results are only the beginning of the ranking. If the correct item is not visible, load 20 more or use manual selection. If the library does not contain the item, mark it as unmatched instead of selecting a merely similar garment.

## Output

Each export creates a separate folder such as `任务001` containing:

- annotated images for confirmed matches;
- unmatched store-photo copies marked in red;
- a visual task overview;
- `任务明细.csv` with per-photo details;
- `补货汇总.csv` with aggregated restock quantities.

Matched results keep the original library filename. Unmatched results keep the original store-photo filename.

## Project Structure

```text
Matching_Clothes/
├─ clothing_matcher_v8_ui_alt/  # Maintained V8 comparison interface
│  ├─ app.py                    # Tkinter desktop UI
│  ├─ engine.py                 # Matching, OCR, color, and export logic
│  ├─ models/                   # Local model directory; weights are not tracked
│  └─ verify_*.py               # Local verification scripts
├─ clothing_matcher_exe/        # Historical versions for reference
├─ clothing-stock-match/        # Codex operational workflow
├─ docs/                        # Chinese user and developer documentation
└─ AGENTS.md                    # Project-level Codex constraints
```

## Data and Privacy

The repository excludes garment libraries, store photos, generated results, model weights, local indexes and feedback, build products, caches, virtual environments, and credentials. V8 settings and caches are stored under `%LOCALAPPDATA%\ClothingMatcherV8`.

Before opening an issue, remove people, store details, orders, local paths, and other sensitive information from screenshots and logs.

## Development and Verification

```powershell
cd clothing_matcher_v8_ui_alt
python -m py_compile app.py engine.py verify_imports.py verify_export.py verify_v8.py
python verify_v8.py
python verify_imports.py
python verify_export.py
```

The full verification scripts use local test libraries or sample photos that are intentionally not included in the public repository. See the Chinese [development and packaging guide](docs/开发与打包.md) for architecture and release details.

## Accuracy Boundaries

- The model cannot find an item that is absent from the selected library.
- Heavy occlusion, extreme cropping, or large annotations over library images may lower the correct result's rank.
- Candidate ranking narrows the manual search space; it does not replace final human review.
- Local feedback is isolated by library path and is not treated as truth for a different library.

## Contributing and Security

Read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting an issue or pull request. Report security or privacy concerns according to [SECURITY.md](SECURITY.md).

## License

No open-source license has been selected yet. Public visibility allows the source to be viewed but does not automatically grant permission to copy, modify, or redistribute it. Contact the repository owner before adoption, commercial use, or redistribution.
