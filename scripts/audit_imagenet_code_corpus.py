from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def count_unique_codes(codes: np.ndarray, codebook_size: int = 16384) -> int:
    # Avoid repeated np.unique calls after the complete codebook is observed,
    # but still scan every block so out-of-range codes cannot hide later in the
    # corpus.
    used = np.zeros(codebook_size, dtype=np.bool_)
    for start in range(0, len(codes), 4096):
        block = np.asarray(codes[start : start + 4096], dtype=np.int64)
        if block.size and (block.min() < 0 or block.max() >= codebook_size):
            raise ValueError(
                f"Code values outside [0, {codebook_size}): "
                f"min={block.min()}, max={block.max()}"
            )
        if not np.all(used):
            used[np.unique(block)] = True
    return int(used.sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    codes = np.load(args.code_path / "codes.npy", mmap_mode="r")
    labels = np.load(args.code_path / "labels.npy", mmap_mode="r")
    valid_single = codes.ndim == 3 and codes.shape[1] in (2, 10) and codes.shape[2] in (256, 576)
    valid_dual = codes.ndim == 4 and codes.shape[1:] == (2, 10, 576)
    if not (valid_single or valid_dual):
        raise ValueError(f"Unexpected code shape: {codes.shape}")
    if labels.shape != (codes.shape[0],):
        raise ValueError(f"Unexpected label shape: {labels.shape}")
    counts = np.bincount(labels.astype(np.int64), minlength=1000)
    payload = {
        "code_path": str(args.code_path),
        "codes_shape": list(codes.shape),
        "codes_dtype": str(codes.dtype),
        "labels_shape": list(labels.shape),
        "labels_dtype": str(labels.dtype),
        "sample_count": int(len(labels)),
        "class_count": int(np.count_nonzero(counts)),
        "min_class_count": int(counts.min()),
        "max_class_count": int(counts.max()),
        "unique_codes": count_unique_codes(codes),
        "codes_sha256": sha256(args.code_path / "codes.npy"),
        "labels_sha256": sha256(args.code_path / "labels.npy"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
