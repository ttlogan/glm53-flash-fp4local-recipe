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

## Review pass 1 (2026-10-17) — fixes applied
A byte-diff review confirmed deepseek + qwen lanes and all shared helpers in
`swap-model.sh` are byte-identical to the pre-change backup (PASS). A serving
review found a hard blocker + minor issues in the launcher, all now fixed in
`glm53-fp4local-vllm.sh`:
- FIXED (blocker): the model-dir existence check used the in-container path
  `/root/.cache/huggingface/...` on the HOST (which is `Permission denied`, so the
  launcher always exited). Now resolves the snapshot from `$HF_CACHE` (host) for the
  check and maps to `/root/.cache/huggingface/...` only for the container vllm args.
- FIXED: added `--dtype bfloat16` (card + recipe had it).
- FIXED: dropped `--decode-context-parallel-size 2` (not in the validated launch;
  risky on 2 GPUs). Defaults to 1; the card's verbose variant used 2, treat as
  optional tuning to test at the gate.
- FIXED: `PORT`/`NAME`/`MPORT` are now env-overridable so the gate smoke test can
  run on `PORT=8001` (and a distinct `NAME`/`MPORT`) without touching qwen on :8000.
- `MODEL_DIR` in `swap-model.sh` is a dead (documentation-only) variable; left as-is.
- `--default-chat-template-kwargs {"reasoning_effort":"max","clear_thinking":false}`
  matches the card's verbose 2x-Spark command (old GLM lane used
  `{"enable_thinking":false}`); confirm intended reasoning behavior at the gate.

Still to verify at the gate: fresh-image support for the QAD scale format, shard
count vs `model.safetensors.index.json` (mid-download during this review), the 1M-ctx
KV fit at `gpu-memory-utilization 0.87` on the 2x128G pool, `--distributed-executor-backend mp`
(whether the B12X image's `mp` path works for 2-node, vs the deepseek lane's Ray),
and whether `/opt/lil/runtime/templates/glm53-flash.jinja` vs the checkpoint's own
`chat_template.jinja` is correct.

## Review pass 2 (2026-10-17) — expert team (3 reviewers)
Verdict: serving flags confirmed valid; orchestration CONCERNS; ops INVARIANTS PASS
(qwen/deepseek byte-preserved). Additional fixes applied:
- FIXED (`ops/swap-model.sh` stop_stack): the final `:8000` check was
  `ss | grep -q && { exit 1; }`, which returns nonzero when `:8000` is ALREADY free
  (grep exit 1) and, under `set -e`, aborted a `--swap`/restart mid-teardown after
  qwen was stopped. Rewritten as `if ...; then ...; fi` so a free port is a normal
  return, not an abort. This was pre-existing but is the one real hazard on the glm
  swap path (would strand `:8000` down).
- FIXED (`glm53-fp4local-vllm.sh`): model-dir resolution now pins to the exact
  `$REVISION` snapshot (was `ls | head -1`, arbitrary if multiple snapshots) and
  verifies every weight shard is present via `model.safetensors.index.json`
  (config.json lands before the weights, so the old guard passed on a partial
  download).
- FIXED (`recipes/glm-5.3-fp4-local.yaml`): removed `kv_cache_memory_bytes: 8G`
  (in this fork it OVERRIDES `gpu_memory_utilization`, capping KV below 1M) — now
  uses only `gpu_memory_utilization 0.87`; set `container` to the actual
  `docker.io/eugr/spark-vllm-b12x:latest`; added `--served-model-name dgx_hobo_default`.

## Deploy-blocking (do before the live GLM swap)
- `/etc/litellm/config.yaml` maps `dgx_hobo_default` -> `openai/qwen3.8-flash-next-uncensored`
  (api_base http://127.0.0.1:8000/v1) with NO `contextWindow`/max_tokens override. On
  the GLM swap the :8000 backend serves name `dgx_hobo_default` (not the qwen model
  name) and litellm would infer the 1M context as qwen's 262K + send the stale
  upstream model name. Update name + `contextWindow` to 1M at swap time. Also update
  the served-name mapping so `dgx_hobo_default` resolves to the GLM backend.
- Stale references to the deleted `/models/glm-5.3-flash-nvfp4` / `/models/dflash2-draft`
  (harmless now but clean up before enabling): disabled `glm53-vllm.service` (leo+raph,
  ExecStart=/home/wfoster/glm53-vllm.sh), `/usr/local/bin/launch-glm.sh`,
  `/home/wfoster/glm53-vllm.sh`, `/home/wfoster/glm53-vllm-dflash2.sh`,
  `/home/wfoster/update-motd-model.sh`, `/opt/glm53/update-motd-model.sh`,
  `/home/wfoster/qwen38-flash-next-recipe/scripts/swap-model.sh`. Active boot unit is
  `dgx-model.service` (runs the clean swap-model.sh), which is unaffected.
