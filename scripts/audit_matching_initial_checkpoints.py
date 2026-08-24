from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    left = torch.load(args.left, map_location="cpu", weights_only=False)
    right = torch.load(args.right, map_location="cpu", weights_only=False)
    left_state = left["model"]
    right_state = right["model"]
    keys_match = left_state.keys() == right_state.keys()
    differing = []
    if keys_match:
        differing = [name for name in left_state if not torch.equal(left_state[name], right_state[name])]
    left_hash = state_sha256(left_state)
    right_hash = state_sha256(right_state)
    result = {
        "left": str(args.left),
        "right": str(args.right),
        "left_steps": int(left["steps"]),
        "right_steps": int(right["steps"]),
        "parameter_tensors": len(left_state),
        "keys_match": keys_match,
        "differing_tensors": differing,
        "left_model_sha256": left_hash,
        "right_model_sha256": right_hash,
        "exact_match": keys_match and not differing and left_hash == right_hash,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["exact_match"]:
        raise SystemExit("Initial checkpoints do not match exactly")


if __name__ == "__main__":
    main()
