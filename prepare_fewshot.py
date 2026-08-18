"""Create a deterministic n-shot ImageFolder subset."""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare an n-shot dataset")
    parser.add_argument("--source", required=True, help="ImageFolder source directory")
    parser.add_argument("--output", required=True, help="new subset directory")
    parser.add_argument("--shots", type=int, required=True, help="images per class")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"source directory does not exist: {source}")
    if args.shots <= 0:
        raise ValueError("shots must be positive")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")

    rng = random.Random(args.seed)
    manifest: list[tuple[str, str]] = []
    for class_dir in sorted(path for path in source.iterdir() if path.is_dir()):
        candidates = [
            path
            for path in sorted(class_dir.rglob("*"))
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if len(candidates) < args.shots:
            raise ValueError(
                f"class {class_dir.name!r} has {len(candidates)} images, "
                f"fewer than --shots={args.shots}"
            )
        selected = rng.sample(candidates, args.shots)
        target_dir = output / class_dir.name
        target_dir.mkdir(parents=True, exist_ok=True)
        for index, source_path in enumerate(selected):
            target_name = f"{index:04d}_{source_path.name}"
            target_path = target_dir / target_name
            shutil.copy2(source_path, target_path)
            manifest.append((class_dir.name, str(source_path.relative_to(source))))

    output.mkdir(parents=True, exist_ok=True)
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "source_path"])
        writer.writerows(manifest)
    print(f"Created {len(manifest)} images in {output}")


if __name__ == "__main__":
    main()
