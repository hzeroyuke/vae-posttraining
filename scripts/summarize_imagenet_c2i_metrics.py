from __future__ import annotations

import argparse
import json
import re
import tempfile
import os
from pathlib import Path


METRIC_PATTERN = re.compile(
    r"^(Inception Score|FID|sFID|Precision|Recall):\s+([-+0-9.eE]+)\s*$"
)


def parse_metrics(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = METRIC_PATTERN.match(line.strip())
        if match:
            values[match.group(1)] = float(match.group(2))
    expected = {"Inception Score", "FID", "sFID", "Precision", "Recall"}
    if values.keys() != expected:
        missing = sorted(expected - values.keys())
        raise ValueError(f"Missing metrics in {path}: {missing}")
    return values


def atomic_json_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    systems = {}
    for name in ("official", "reward"):
        root = args.evaluation_root / name
        systems[name] = {
            "manifest": json.loads((root / "manifest.json").read_text(encoding="utf-8")),
            "metrics": parse_metrics(root / "metrics.txt"),
        }
    protocol_names = (
        "gpt_model",
        "gpt_type",
        "image_size",
        "image_size_eval",
        "num_classes",
        "cfg_scale",
        "cfg_interval",
        "temperature",
        "top_k",
        "top_p",
        "num_fid_samples",
        "global_seed",
        "world_size",
        "per_rank_batch_size",
        "sampling_scheme",
    )
    official_protocol = {
        name: systems["official"]["manifest"][name] for name in protocol_names
    }
    reward_protocol = {
        name: systems["reward"]["manifest"][name] for name in protocol_names
    }
    if official_protocol != reward_protocol:
        raise ValueError("Official/reward sampling protocols differ")
    result = {
        "reference": str(args.reference.resolve()),
        "protocol": official_protocol,
        "official_reference": {
            "published_xl_fid": 2.629,
            "published_training_epochs": 300,
            "cfg_scale": 1.75,
        },
        "systems": systems,
        "reward_minus_official": {
            name: systems["reward"]["metrics"][name]
            - systems["official"]["metrics"][name]
            for name in systems["official"]["metrics"]
        },
    }
    atomic_json_write(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
