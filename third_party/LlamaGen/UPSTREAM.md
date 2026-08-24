# LlamaGen source

The files in this directory are the minimal dependency subset used by this experiment.
They come from FoundationVision/LlamaGen at upstream commit
`ce98ec4` plus the local COCO sampled-code,
T5-shard, dataset, and matched-prior training additions used for the recorded run.

The upstream license is retained in `LICENSE`.

The ImageNet c2i terminal evaluation also vendors the upstream
`evaluations/c2i/evaluator.py` unchanged from the same commit. It implements the
OpenAI guided-diffusion FID/sFID/Inception Score/Precision/Recall protocol used by
the LlamaGen release.
