# GLM-5.3-Flash-FP4-Local — findings & gate items

## Context
Swapping the `glm` lane on the 2-node GX10 cluster from `RedHatAI/GLM-5.3-Flash-NVFP4`
(LLM-Compressor PTQ, marlin/DFlash2) to `nvidia/GLM-5.3-Flash-FP4-Local` (ModelOpt
QAD, B12X/MTP). The RedHatAI build was a plain PTQ quant of the same base; the
NVIDIA QAD build is the same base, still not uncensored, but higher-quality FP4.

## What the model card reports (nvidia/GLM-5.3-Flash-FP4-Local)
- 320B total / 18B active, `Glm5NextForConditionalGeneration`, text/image/video in,
  text out. Vision stack is quantized (MXFP8 attention linears, NVFP4 MLP/merger).
- QAD with nvidia-modelopt v0.47.0; trained 500M tokens x3 epochs.
- Benchmarks (vs BF16 base): GPQA Diamond 91.3% (BF16 92.1%), ToolEvalBench 92.4%
  (BF16 91.9%). Near-lossless, and better than BF16 on tool-use.
- ~166 GiB vRAM. Validated on 2x DGX Spark (2M tokens FP8 KV cache) and 2x RTX PRO 6000.
- License: NVIDIA Open Model Agreement (base is MIT). NOT gated on HF (verified: gated=False).
- **Critical caveat from the card:** the NVFP4 scale factors are a NEW lossless
  compression format that upstream vLLM does not support yet; you MUST use one of
  the provided community containers (`eugr/spark-vllm-b12x:latest` for Spark,
  `ghcr.io/local-inference-lab/vllm:release-10152026` for RTX PRO).

## Cluster facts measured on 2026-10-17
- Single 916G ext4 filesystem on each node (`/models` is a dir on `/`, not a mount).
  leo was at 100% (6.8G free), raph at 98% (20G free).
- Old GLM weights `/models/glm-5.3-flash-nvfp4` = 185G; `/models/dflash2-draft` = 2.2G.
- Both were deleted (plus the two `tonyd2wild/vllm-glm53-flash` images, 63G) to fit
  the ~166G nvidia checkpoint. DeepSeek weights (160G) and qwen kept.
- The nvidia checkpoint is downloaded into wfoster's HF cache on BOTH nodes
  (`/home/wfoster/.cache/huggingface/hub/models--nvidia--GLM-5.3-Flash-FP4-Local`,
  revision `76348ddb`). raph has no `hf`/huggingface_cli on the host, so it downloaded
  via the `eugr/spark-vllm-b12x:latest` container (huggingface_hub 1.33.0).
- `docker.io/eugr/spark-vllm-b12x:latest` is present on both nodes.
- Deployed `model-tui` = `/home/wfoster/model-tui/model-tui.py` (venv wrapper at
  `/usr/local/bin/model-tui`). `swap-model.sh` = `/usr/local/bin/swap-model.sh`.

## Staged orchestration changes
- New launcher `glm53-fp4local-vllm.sh` (deployed to `/home/wfoster/` on both nodes):
  image `eugr/spark-vllm-b12x:latest`, B12X/moe/linear backends, `modelopt_mixed`,
  `load-format instanttensor`, `mamba-cache-mode align`, `kv-cache-dtype fp8`,
  `block-size 256`, `max-model-len 1048576`, native MTP spec-decode (no DFlash2
  drafter), served name `dgx_hobo_default`, master port :29521 (head-first).
- `swap-model.sh`: `glm` lane now points at `glm53-fp4local-vllm.sh`; MODEL_DIR ->
  HF cache path. deepseek/qwen branches untouched.
- `model-tui.py`: `glm` row display updated to `GLM-5.3-Flash-FP4-Local (b12x)`.

## GATE ITEMS (must validate before the live swap)
1. **Fresh image.** The cached `eugr/spark-vllm-b12x:latest` on the cluster is built
   `0.1.dev21553+gab86b7073.d20261002` (2026-10-02), BEFORE the model (released
   2026-10-05). It registers `modelopt_mixed` with E4M3 scales (looks right), but
   confirm it decodes the QAD checkpoint's new scale format. If not, pull a fresh
   `:latest` or build `local-inference-lab/vllm` to a release >= the model.
2. **Load + smoke test on a SEPARATE port (:8001)** so qwen (on :8000) is untouched.
   Head on leo, worker on raph, poll `/health`.
3. **Fabric env reconciliation.** This recipe assumes the cluster's proven b12x fabric
   (NCCL_NET=Socket, NCCL_IB_DISABLE=1). The card also ships a RoCE-twin variant
   (NCCL_IB_DISABLE=0, VLLM_ENABLE_ROCE_ALLREDUCE=1, IB_IF=rocep1s0f0,roceP2p1s0f0).
   Record which actually works on this fabric.
4. **load-format** `instanttensor` vs `b12x` (the card's Spark block uses instanttensor;
   the eugr recipe uses b12x). Validate both; note any INSTANTTENSOR_* env that's required.
5. **chat template.** The card points at `/opt/lil/runtime/templates/glm53-flash.jinja`,
   but that dir is empty in the cached image; the launcher uses the checkpoint's own
   `chat_template.jinja` instead.
6. **Vision + multimodal path** confirmed active on the B12X image.
7. **1M ctx KV fit.** The card's Spark example uses `gpu-memory-utilization 0.87`,
   `max-model-len 1048576`; confirm it doesn't OOM on the shared 2x128G unified pool
   and matches the desired 1M ctx.
8. Since the base model is unchanged, confirm the served name / omp `dgx_hobo_default`
   contextWindow stays consistent with 1M (no client churn).

## Rollback
Re-download `RedHatAI/GLM-5.3-Flash-NVFP4` (or `local-inference-lab/GLM-5.3-Flash-NVFP4`)
and re-pull `ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v8` to restore the old lane.
