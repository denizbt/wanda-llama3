#!/usr/bin/env python3
"""Export Hugging Face ImageNet-1k into torchvision ImageFolder directories."""

import argparse
import os
import sys
from pathlib import Path

# This directory also contains datasets.py for the image-classifier loader.
# Remove it from import lookup so `datasets` below resolves to Hugging Face's
# installed package instead of that local module.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [
    entry
    for entry in sys.path
    if Path(entry or os.getcwd()).resolve() != SCRIPT_DIR
]

from datasets import Image, load_dataset


def export_split(
    hf_split: str,
    output_name: str,
    output_root: Path,
    max_examples: int | None = None,
) -> None:
    split_dir = output_root / output_name
    complete_marker = split_dir / ".complete"
    if complete_marker.exists():
        print(f"Skipping completed split: {split_dir}")
        return
    if max_examples is not None and split_dir.exists():
        existing = sum(1 for path in split_dir.glob("*/*") if path.is_file())
        if existing >= max_examples:
            print(f"Reusing {existing:,} diagnostic images in {split_dir}")
            return

    split_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("ILSVRC/imagenet-1k", split=hf_split, streaming=True)
    dataset = dataset.cast_column("image", Image(decode=False))

    count = 0
    for index, example in enumerate(dataset):
        if max_examples is not None and count >= max_examples:
            break

        label = int(example["label"])
        if label < 0:
            continue

        class_dir = split_dir / f"{label:04d}"
        class_dir.mkdir(exist_ok=True)

        image = example["image"]
        image_bytes = image.get("bytes")
        source_path = image.get("path")
        if image_bytes is None and source_path:
            with open(source_path, "rb") as source:
                image_bytes = source.read()
        if image_bytes is None:
            raise RuntimeError(f"No encoded image data for {hf_split} example {index}")

        suffix = Path(source_path or "image.JPEG").suffix or ".JPEG"
        destination = class_dir / f"{index:08d}{suffix}"
        if not destination.exists():
            temporary = destination.with_suffix(destination.suffix + ".part")
            with open(temporary, "wb") as output:
                output.write(image_bytes)
            os.replace(temporary, destination)

        count += 1
        if count % 10_000 == 0:
            print(f"Exported {count:,} images to {split_dir}", flush=True)

    if max_examples is None:
        complete_marker.touch()
    print(f"Completed {split_dir}: {count:,} images")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--max-examples-per-split",
        type=int,
        default=None,
        help="export only this many labeled examples from each split",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    export_split("train", "train", args.output_dir, args.max_examples_per_split)
    export_split("validation", "val", args.output_dir, args.max_examples_per_split)


if __name__ == "__main__":
    main()
