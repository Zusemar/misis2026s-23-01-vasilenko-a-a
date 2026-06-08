#!/usr/bin/env python3
"""Reduce every image in a folder to a fixed number of colors with K-means."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def quantize_colors(image: np.ndarray, color_count: int, seed: int) -> np.ndarray:
    pixels = image.reshape(-1, 3).astype(np.float32)
    unique_colors = np.unique(pixels, axis=0)
    if len(unique_colors) <= color_count:
        return image.copy()

    cv2.setRNGSeed(seed)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(
        pixels,
        color_count,
        None,
        criteria,
        5,
        cv2.KMEANS_PP_CENTERS,
    )
    centers = np.clip(np.rint(centers), 0, 255).astype(np.uint8)
    return centers[labels.ravel()].reshape(image.shape)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", nargs="?", type=Path, default=Path("result"))
    parser.add_argument("output_dir", nargs="?", type=Path, default=Path("кластеризация"))
    parser.add_argument("--colors", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    image_paths = sorted(
        path
        for path in args.input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise SystemExit(f"No images found in {args.input_dir}")

    if args.output_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"Output directory already exists: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    for index, path in enumerate(image_paths):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise SystemExit(f"Could not read image: {path}")
        quantized = quantize_colors(image, args.colors, seed=42 + index)
        if not cv2.imwrite(str(args.output_dir / path.name), quantized):
            raise SystemExit(f"Could not write image: {args.output_dir / path.name}")
        print(f"[{index + 1:02d}/{len(image_paths):02d}] {path.name}")


if __name__ == "__main__":
    main()
