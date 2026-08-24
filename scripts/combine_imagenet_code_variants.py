from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def read_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, required=True, help="crop-range 1.10 codes")
    parser.add_argument("--secondary", type=Path, required=True, help="crop-range 1.05 codes")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=1024)
    args = parser.parse_args()

    primary_codes = np.load(args.primary / "codes.npy", mmap_mode="r")
    secondary_codes = np.load(args.secondary / "codes.npy", mmap_mode="r")
    primary_labels = np.load(args.primary / "labels.npy", mmap_mode="r")
    secondary_labels = np.load(args.secondary / "labels.npy", mmap_mode="r")
    primary_manifest = read_json(args.primary / "manifest.json")
    secondary_manifest = read_json(args.secondary / "manifest.json")

    if primary_codes.shape != secondary_codes.shape:
        raise ValueError(
            f"Code shapes differ: {primary_codes.shape} != {secondary_codes.shape}"
        )
    if primary_codes.ndim != 3 or primary_codes.shape[1:] != (10, 576):
        raise ValueError(f"Expected [N,10,576] source codes, got {primary_codes.shape}")
    if primary_codes.dtype != secondary_codes.dtype:
        raise ValueError(
            f"Code dtypes differ: {primary_codes.dtype} != {secondary_codes.dtype}"
        )
    if primary_labels.shape != secondary_labels.shape or primary_labels.shape != (
        primary_codes.shape[0],
    ):
        raise ValueError("Source label shapes do not match the code arrays")
    if primary_labels.dtype != secondary_labels.dtype:
        raise ValueError(
            f"Label dtypes differ: {primary_labels.dtype} != {secondary_labels.dtype}"
        )
    if float(primary_manifest["crop_range"]) != 1.1:
        raise ValueError("Primary manifest is not crop_range=1.1")
    if float(secondary_manifest["crop_range"]) != 1.05:
        raise ValueError("Secondary manifest is not crop_range=1.05")
    if (
        primary_manifest["vq_checkpoint_sha256"]
        != secondary_manifest["vq_checkpoint_sha256"]
    ):
        raise ValueError("Crop ranges were not encoded by the same VQ checkpoint")

    args.output.mkdir(parents=True, exist_ok=False)
    codes_temporary = args.output / ".codes.npy.tmp"
    labels_temporary = args.output / ".labels.npy.tmp"
    output_codes = np.lib.format.open_memmap(
        codes_temporary,
        mode="w+",
        dtype=primary_codes.dtype,
        shape=(primary_codes.shape[0], 2, 10, 576),
    )
    output_labels = np.lib.format.open_memmap(
        labels_temporary,
        mode="w+",
        dtype=primary_labels.dtype,
        shape=primary_labels.shape,
    )
    for start in range(0, len(primary_codes), args.chunk_size):
        stop = min(start + args.chunk_size, len(primary_codes))
        first_labels = np.asarray(primary_labels[start:stop])
        second_labels = np.asarray(secondary_labels[start:stop])
        if not np.array_equal(first_labels, second_labels):
            raise ValueError(f"Labels differ in rows [{start}:{stop})")
        output_codes[start:stop, 0] = primary_codes[start:stop]
        output_codes[start:stop, 1] = secondary_codes[start:stop]
        output_labels[start:stop] = first_labels
        if stop % 100_000 == 0 or stop == len(primary_codes):
            print(f"combined {stop:,}/{len(primary_codes):,}", flush=True)
    output_codes.flush()
    output_labels.flush()
    del output_codes, output_labels
    os.replace(codes_temporary, args.output / "codes.npy")
    os.replace(labels_temporary, args.output / "labels.npy")

    manifest = {
        "data_path": primary_manifest["data_path"],
        "split": primary_manifest["split"],
        "sample_count": int(primary_codes.shape[0]),
        "augmentation_groups": 2,
        "augmentations_per_group": 10,
        "augmentations_per_image": 20,
        "tokens_per_image": 576,
        "image_size": 384,
        "ten_crop": True,
        "crop_ranges": [1.1, 1.05],
        "vq_checkpoint": primary_manifest["vq_checkpoint"],
        "vq_checkpoint_sha256": primary_manifest["vq_checkpoint_sha256"],
        "global_seed": primary_manifest["global_seed"],
        "variant_paths": [str(args.primary.resolve()), str(args.secondary.resolve())],
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
