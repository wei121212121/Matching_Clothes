from pathlib import Path
import time

from engine import EmbeddingEngine, LibraryIndex


ROOT = Path(__file__).resolve().parent.parent
LIBRARY = ROOT / "2026短袖"
TEMP = Path.home() / "AppData" / "Local" / "Temp"

TESTS = [
    ("codex-clipboard-cb0fc386-fa1a-41a4-bc63-0ae47c9d024f.jpg", "25900_大创_1.png"),
    ("codex-clipboard-9933f5de-1281-429f-9a42-081c8f861473.jpg", "22900_至尊_7.png"),
    ("codex-clipboard-bab8273b-a9a9-42c4-b5b4-98e4bb35594c.jpg", "22900_至尊_21.png"),
    ("codex-clipboard-db4500ae-62dc-4920-9f64-48324c543c68.jpg", "22900_至尊_19.png"),
]


def main() -> None:
    engine = EmbeddingEngine(None)
    index = LibraryIndex(LIBRARY, engine)
    started = time.time()
    count = index.build()
    print(f"indexed={count} seconds={time.time() - started:.2f} mode={engine.mode}")
    for filename, expected in TESTS:
        path = TEMP / filename
        if not path.exists():
            print(f"missing={path}")
            continue
        started = time.time()
        matches = index.search(path, top_k=10)
        names = [match.path.name for match in matches]
        rank = names.index(expected) + 1 if expected in names else None
        print(f"\ninput={filename} expected={expected} rank={rank} seconds={time.time() - started:.2f}")
        for position, match in enumerate(matches[:5], start=1):
            print(f"{position}: {match.score:.4f} {match.path.name}")


if __name__ == "__main__":
    main()
