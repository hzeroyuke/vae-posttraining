from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import torch


EXPECTED_CORPUS = {
    "sample_count": 1_281_167,
    "codes_shape": [1_281_167, 2, 10, 576],
    "labels_shape": [1_281_167],
    "class_count": 1_000,
    "unique_codes": 16_384,
}

EXPECTED_PROTOCOL = {
    "dataset": "imagenet_code",
    "gpt_model": "GPT-XL",
    "gpt_type": "c2i",
    "image_size": 384,
    "downsample_size": 16,
    "num_classes": 1_000,
    "vocab_size": 16_384,
    "cls_token_num": 1,
    "epochs": 300,
    "max_steps": 78_000,
    "lr": 1e-4,
    "weight_decay": 0.05,
    "beta1": 0.9,
    "beta2": 0.95,
    "max_grad_norm": 1.0,
    "dropout_p": 0.1,
    "token_dropout_p": 0.1,
    "drop_path_rate": 0.0,
    "global_batch_size": 256,
    "global_seed": 20_260_812,
    "mixed_precision": "bf16",
    "ema": False,
    "no_compile": True,
    "preload_epoch_codes": True,
    "ckpt_every": 10_000,
}

RUNTIME_GEOMETRY = ("world_size", "gradient_accumulation_steps")

OFFICIAL_REFERENCE = {
    "source": "https://arxiv.org/abs/2406.06525",
    "epochs": 300,
    "reported_xl_fid": 2.629,
    "cfg_scale": 1.75,
}


def updates_per_epoch(world_size: int, accumulation: int) -> int:
    global_batch = int(EXPECTED_PROTOCOL["global_batch_size"])
    per_rank_batch = global_batch // (world_size * accumulation)
    per_rank_samples = (int(EXPECTED_CORPUS["sample_count"]) + world_size - 1) // world_size
    micro_batches = per_rank_samples // per_rank_batch
    return micro_batches // accumulation


def state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def namespace_values(value: object, names: object) -> dict[str, object]:
    return {name: getattr(value, name, None) for name in names}


def audit_checkpoint(path: Path, expected_steps: int, expected_code_path: Path) -> dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    state = checkpoint["model"]
    steps = int(checkpoint["steps"])
    parameter_count = sum(int(value.numel()) for value in state.values())
    if steps != expected_steps:
        raise ValueError(f"{path}: expected steps={expected_steps}, got {steps}")
    if parameter_count != 774_699_520:
        raise ValueError(f"{path}: unexpected parameter count {parameter_count}")
    protocol = namespace_values(checkpoint["args"], EXPECTED_PROTOCOL)
    protocol.update(namespace_values(checkpoint["args"], RUNTIME_GEOMETRY))
    expected_protocol = {**EXPECTED_PROTOCOL, "max_steps": expected_steps}
    observed_common = {name: protocol[name] for name in EXPECTED_PROTOCOL}
    if observed_common != expected_protocol:
        raise ValueError(
            f"{path}: protocol mismatch: {observed_common} != {expected_protocol}"
        )
    world_size = int(protocol["world_size"])
    accumulation = int(protocol["gradient_accumulation_steps"])
    global_batch = int(protocol["global_batch_size"])
    if world_size < 1 or accumulation < 1 or global_batch % (world_size * accumulation):
        raise ValueError(
            f"{path}: invalid runtime geometry: world_size={world_size}, "
            f"gradient_accumulation_steps={accumulation}, global_batch_size={global_batch}"
        )
    code_path = Path(checkpoint["args"].code_path).resolve()
    if code_path != expected_code_path.resolve():
        raise ValueError(f"{path}: unexpected code path {code_path}")
    optimizer_state_count = len(checkpoint["optimizer"]["state"])
    if optimizer_state_count == 0:
        raise ValueError(f"{path}: optimizer state is empty")
    signature = {
        name: (str(value.dtype), list(value.shape))
        for name, value in sorted(state.items())
    }
    signature_hash = hashlib.sha256(
        json.dumps(signature, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "steps": steps,
        "parameter_tensors": len(state),
        "parameter_count": parameter_count,
        "optimizer_state_count": optimizer_state_count,
        "protocol": protocol,
        "code_path": str(code_path),
        "state_signature_sha256": signature_hash,
        "model_sha256": state_sha256(state),
    }


def read_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return value


def audit_corpus(path: Path) -> dict[str, object]:
    manifest = read_json(path / "manifest.json")
    audit = read_json(path / "audit.json")
    observed = {name: audit[name] for name in EXPECTED_CORPUS}
    if observed != EXPECTED_CORPUS:
        raise ValueError(f"{path}: corpus mismatch: {observed} != {EXPECTED_CORPUS}")
    expected_manifest = {
        "sample_count": EXPECTED_CORPUS["sample_count"],
        "augmentation_groups": 2,
        "augmentations_per_group": 10,
        "augmentations_per_image": 20,
        "tokens_per_image": 576,
        "image_size": 384,
        "ten_crop": True,
        "crop_ranges": [1.1, 1.05],
    }
    manifest_protocol = {name: manifest[name] for name in expected_manifest}
    if manifest_protocol != expected_manifest:
        raise ValueError(
            f"{path}: manifest mismatch: {manifest_protocol} != {expected_manifest}"
        )
    return {
        **observed,
        **manifest_protocol,
        "codes_dtype": audit["codes_dtype"],
        "labels_dtype": audit["labels_dtype"],
        "codes_sha256": audit["codes_sha256"],
        "labels_sha256": audit["labels_sha256"],
        "vq_checkpoint": manifest["vq_checkpoint"],
        "vq_checkpoint_sha256": manifest["vq_checkpoint_sha256"],
    }


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
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int, default=78000)
    args = parser.parse_args()
    official_codes = args.experiment_root / "codes/official"
    reward_codes = args.experiment_root / "codes/reward"
    official = args.experiment_root / "training/official/results/000-GPT-XL/checkpoints" / f"{args.expected_steps:07d}.pt"
    reward = args.experiment_root / "training/reward/results/000-GPT-XL/checkpoints" / f"{args.expected_steps:07d}.pt"
    initial = read_json(args.experiment_root / "initial_checkpoint_audit.json")
    if not initial.get("exact_match") or initial.get("left_steps") != 0 or initial.get("right_steps") != 0:
        raise ValueError("Step-0 checkpoint audit is not an exact match")
    official_audit = audit_checkpoint(official, args.expected_steps, official_codes)
    reward_audit = audit_checkpoint(reward, args.expected_steps, reward_codes)
    epoch_updates = updates_per_epoch(
        int(reward_audit["protocol"]["world_size"]),
        int(reward_audit["protocol"]["gradient_accumulation_steps"]),
    )
    official_reference_steps = epoch_updates * int(OFFICIAL_REFERENCE["epochs"])
    result = {
        "expected_steps": args.expected_steps,
        "endpoint_context": {
            "updates_per_epoch": epoch_updates,
            "matched_endpoint_steps": args.expected_steps,
            "matched_endpoint_equivalent_epochs": args.expected_steps / epoch_updates,
            "official_reference": {
                **OFFICIAL_REFERENCE,
                "equivalent_optimizer_steps": official_reference_steps,
            },
            "matches_official_epoch_budget": args.expected_steps == official_reference_steps,
        },
        "corpora": {
            "official": audit_corpus(official_codes),
            "reward": audit_corpus(reward_codes),
        },
        "initial_checkpoint": initial,
        "official": official_audit,
        "reward": reward_audit,
    }
    if result["corpora"]["official"]["labels_sha256"] != result["corpora"]["reward"]["labels_sha256"]:
        raise ValueError("Official/reward label arrays differ")
    if result["corpora"]["official"]["vq_checkpoint_sha256"] == result["corpora"]["reward"]["vq_checkpoint_sha256"]:
        raise ValueError("Official/reward corpora unexpectedly use the same VQ checkpoint")
    if result["official"]["state_signature_sha256"] != result["reward"]["state_signature_sha256"]:
        raise ValueError("Official/reward final checkpoint state signatures differ")
    result["model_sha256_match"] = result["official"]["model_sha256"] == result["reward"]["model_sha256"]
    output = args.experiment_root / "final_checkpoint_audit.json"
    atomic_json_write(output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
