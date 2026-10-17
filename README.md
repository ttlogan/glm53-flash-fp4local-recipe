# GLM-5.3-Flash-FP4-Local (nvidia QAD) — 2x GX10 / GB10 recipe

Serving `nvidia/GLM-5.3-Flash-FP4-Local` on the 2-node ASUS GX10 (GB10 / sm_121a)
cluster, on the B12X vLLM stack with native MTP spec-decode.

## What this is
`nvidia/GLM-5.3-Flash-FP4-Local` is a **quantization-aware-distillation (QAD)** FP4
quant of `zai-org/GLM-5.3-Flash` (Model Optimizer v0.47.0). QAD recovers accuracy
at FP4 better than plain post-training quantization. It is 320B total / 18B active,
multimodal (text/image/video in, text out), and keeps the base model's safety
alignment (it is NOT an abliterated/uncensored build).

## Why the B12X stack
The checkpoint's NVFP4 scale factors use a **new lossless compression format** that
upstream vLLM does not yet read. The model card requires one of the provided
community containers. For DGX Spark / GX10 that is `eugr/spark-vllm-b12x:latest`.
This recipe therefore serves GLM on the B12X stack (attention/moe/linear backends
`b12x`, `--quantization modelopt_mixed`, native MTP), not the old marlin/DFlash2 lane.

## License
`nvidia-open-model-agreement` (base `zai-org/GLM-5.3-Flash` is MIT). Not gated on HF.

## Files
- `glm53-fp4local-vllm.sh` — rank-aware launcher (head-first, master port :29521,
  serves `dgx_hobo_default`, model resolved from wfoster's HF cache). Deployed to
  `/home/wfoster/glm53-fp4local-vllm.sh` on both nodes.
- `recipes/glm-5.3-fp4-local.yaml` — eugr `run-recipe.sh` recipe (B12X image).
- `docs/findings.md` — measured results, the gate items, and what still needs
  validation before going live.

## Run (after the model is cached on both nodes)
```bash
# head first (rank 0), wait for :29521, then worker (rank 1)
sudo -u wfoster bash /home/wfoster/glm53-fp4local-vllm.sh 0
sudo -u wfoster ssh raph "bash /home/wfoster/glm53-fp4local-vllm.sh 1"
```
Live switch (stops qwen): `sudo /usr/local/bin/swap-model.sh --swap glm`.

## Status
Staged, not yet validated. The cached B12X image on the cluster predates the model,
so a fresh image / load test is the gate before going live. See `docs/findings.md`.
