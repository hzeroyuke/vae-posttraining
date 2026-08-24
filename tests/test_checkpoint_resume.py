from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


LLAMAGEN_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "LlamaGen"
sys.path.insert(0, str(LLAMAGEN_ROOT))

from autoregressive.train.checkpoint_resume import resolve_resume_geometry


class CheckpointResumeGeometryTests(unittest.TestCase):
    def test_same_geometry_preserves_micro_steps(self):
        checkpoint = {
            "micro_steps_in_epoch": 159_872,
            "args": SimpleNamespace(
                world_size=2,
                gradient_accumulation_steps=32,
                global_batch_size=256,
            ),
        }

        geometry = resolve_resume_geometry(
            checkpoint,
            fallback_micro_steps=0,
            current_world_size=2,
            current_gradient_accumulation=32,
            current_global_batch_size=256,
        )

        self.assertEqual(geometry.micro_steps, 159_872)
        self.assertEqual(geometry.consumed_samples_in_epoch, 1_278_976)

    def test_two_to_four_gpu_resume_preserves_consumed_samples(self):
        checkpoint = {
            "micro_steps_in_epoch": 159_872,
            "args": SimpleNamespace(
                world_size=2,
                gradient_accumulation_steps=32,
                global_batch_size=256,
            ),
        }

        geometry = resolve_resume_geometry(
            checkpoint,
            fallback_micro_steps=0,
            current_world_size=4,
            current_gradient_accumulation=16,
            current_global_batch_size=256,
        )

        self.assertEqual(geometry.micro_steps, 79_936)
        self.assertEqual(geometry.source_per_rank_batch_size, 4)
        self.assertEqual(geometry.consumed_samples_in_epoch, 1_278_976)

    def test_missing_checkpoint_geometry_uses_current_geometry(self):
        geometry = resolve_resume_geometry(
            {},
            fallback_micro_steps=320,
            current_world_size=4,
            current_gradient_accumulation=16,
            current_global_batch_size=256,
        )

        self.assertEqual(geometry.micro_steps, 320)


if __name__ == "__main__":
    unittest.main()
