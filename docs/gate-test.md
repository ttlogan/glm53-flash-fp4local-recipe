# GLM-5.3-Flash-FP4-Local — Gate Test Log

Live running record of the gate test for `nvidia/GLM-5.3-Flash-FP4-Local` on the
2x GX10 (GB10 / sm_121a) cluster. Append as we go.

## 2026-10-17 — pre-test state
- leo (rank0 / head): checkpoint download had FAILED mid-transfer — `hf download`
  crashed with `Xet CAS Client Error: ... File reconstruction error: I/O error:
  error decoding response body`. 85G / 29 blobs at failure. Restarted the download
  (resumes; skips completed blobs). Disk: 139G free (85% used).
- raph (rank1 / worker): checkpoint download COMPLETE — 166G, 52 blobs, all 52 files
  fetched, `model.safetensors.index.json` present.
- qwen lane (container `vllm-fn`) up on BOTH nodes (Up 29h), served
  `qwen3.8-flash-next-uncensored`, `--master-port 50000`, gpu-memory-utilization 0.80,
  max-model-len 262144. So `:29521` / `:29523` are FREE for the GLM smoke test.
- GPU memory per node: GB10, ~121G total, ~110G used (qwen), ~10G free.
  => GLM (gpu-memory-utilization 0.87, ~80G/node weights) CANNOT coexist with qwen;
  qwen must be stopped for the load test.
- b12x images present on both nodes:
  docker.io/eugr/spark-vllm-b12x:latest (22338a4c37d3, 25G),
  localhost/vllm-node-b12x-vision:latest (9fc9df2ea33a, 24.9G).
- SMOKE test plan: run the launcher on a SCRATCH port/container so the active lane
  (and /etc/vllm/current_model=qwen38) is untouched, then decide the flip:
  PORT=8001, NAME=vllm_glm53_smoke, MPORT=29523, served-model-name dgx_hobo_default.

## Decision needed (open)
- User authorized stopping qwen for the GLM test (OOB: "you need to stop qwen").
- After a successful gate smoke test, decide: bring qwen back, or do the live flip
  (which also requires updating /etc/litellm/config.yaml: dgx_hobo_default -> GLM
  backend, contextWindow -> 1M).

## TBD (append results below as they happen)

## 2026-10-17 — context length (user asked re: "cited for 2M context")
- Checkpoint `config.json`: `max_position_embeddings: 1048576` (1M), `rope_scaling: None`.
- NVIDIA FP4-Local card, Input section: "Context length up to 1M".
- The card's "2 x NVIDIA DGX Spark 128 GB with **2M tokens of FP8 KV Cache**" is the
  KV-CACHE CAPACITY on 2x Spark, NOT the context length. Max context is 1M.
- Card's 2x DGX Spark serving examples use `--max-model-len 500000` (one-step/manual)
  or `1048576` (verbose RTX-PRO/PRO block). Never 2M.
=> Serve at max-model-len 1048576 (1M). A 2M window would need rope scaling the
published checkpoint does not carry. So the launcher/recipe 1M value is correct.

## 2026-10-17 — card has TWO 2x DGX Spark serving variants (resolve at gate)
- Variant A (eugr recipe / one-step launcher `recipes/glm-5.3-flash.yaml` + verbose):
  `--load-format instanttensor`, `--max-model-len 1048576`, spec
  moe_backend=b12x num_spec_tokens=3, env INSTANTTENSOR_* (BUFFERED/64M...),
  fabric Socket (NCCL_NET=Socket, NCCL_IB_DISABLE=1). This is what the current
  launcher/recipe use.
- Variant B (card "manual launch", eugr container):
  `--load-format b12x`, `--max-model-len 500000`, spec moe_backend=**humming**
  num_spec_tokens=**5**, env VLLM_USE_AOT_COMPILE=1, VLLM_USE_MEGA_AOT_ARTIFACT=1,
  VLLM_SSM_CONV_STATE_LAYOUT=DS, VLLM_ENABLE_PCIE_ALLREDUCE=0, fabric RoCE
  (NCCL_IB_DISABLE=0, VLLM_ENABLE_ROCE_ALLREDUCE=1, IB_IF=rocep1s0f0,roceP2p1s0f0),
  gpu-memory-utilization 0.87.
- Decision for gate: first attempt Variant A (matches the eugr one-step launcher the
  card points to, and the cluster's proven deepseek Socket fabric). If the load fails
  on quantization/load-format, fall back to Variant B (b12x / humming / 500K) and its
  RoCE fabric; note the cluster's proven fabric is Socket, so RoCE ("twins" + separate
  subnets + workers-before-head) would be a new, untested path.

## 2026-10-17 — model-tui ACTIVE-clears fix (user reported after --stop)
- Bug (pre-existing, flagged in review pass 1/2): `active_model()` trusted
  `/etc/vllm/current_model` (the boot DEFAULT = last-used), so after a `--stop`
  the stopped model still showed the green ACTIVE dot though :8000 was free.
- Fix: `active_model()` now probes `http://127.0.0.1:8000/v1/models` and matches the
  served model name back to a lane via a new `SERVED` map (sync with
  `swap-model.sh served_name()`); if nothing is listening it returns None and the
  ACTIVE dot clears. Trimmed the now-unused `STATE_FILE`.
- Verified on leo with qwen down: `_serving_lane()` / `active_model()` return None.
  Deployed, committed, pushed (954b2db).
- Note: `SERVED` must stay in sync with `swap-model.sh served_name()` (deepseek=
  orcarouter-deepseek-v4-flash-vision-uncensored, glm=dgx_hobo_default,
  qwen38=qwen3.8-flash-next-uncensored).

## 2026-10-17 — download monitor (continued)
- leo: 130G -> 142G -> 154G -> 160G / 166G, 52/52 blobs, index.json present,
  4 in-flight, hf download alive. ~6G to go; launch once 0 incomplete.
- qwen STOPPED (per user via model-tui) on both nodes; :8000 free; leo GPU ~112G free.

## Known / deferred (not fixing now)
- model-tui live-log pane (`ProgressScreen` 'l' toggles `podman logs -f`): user reports
  it "has never worked right." Deferred; not part of this GLM gate.

## 2026-10-17 — GLM smoke test: blocker diagnosed (IMAGE FRESHNESS)
Repeated launch attempts on scratch :8001/29523. Two distinct failures, root cause finally
isolated:
1. (load-format instanttensor) `FileNotFoundError: Weight files referenced in index but
   missing: .../tensors/native-QB.safetensors`. The checkpoint uses a NON-standard QAD
   layout (scales in `tensors/native-QB.safetensors`, 719M, present on both nodes).
   -> fix: `--load-format b12x` (card's DGX Spark value) which routes via B12xModelLoader.
2. (still failing with b12x) `b12x weight loading failed: 0/0 shards routed, 0.00 GB
   selected on rank 0`, then `RuntimeError: Engine core initialization failed`. vLLM
   instantiated the OLD class `glm4_1v.py`, NOT a GLM-5.3 path.
   ROOT CAUSE: the cached `eugr/spark-vllm-b12x:latest` image (v0.1.dev21553+...d20261002,
   created 2026-10-02, digest d7213eea...) predates the 2026-10-05 model. Its model
   registry maps `Glm53DSparkForCausalLM -> glm53_dspark` (the OLDER
   `local-inference-lab/GLM-5.3-Flash-NVFP4-Spark` checkpoint) but NOT the nvidia
   checkpoint's `model_type=glm5_next` / `Glm5NextForConditionalGeneration`. So vLLM
   can't resolve the right class and the b12x loader routes 0 shards.
   => The image must be refreshed to a build that registers Glm5NextForConditionalGeneration.
   (Note: the eugr recipe recipes/glm-5.3-flash.yaml targets the DIFFERENT
   `local-inference-lab/GLM-5.3-Flash-NVFP4-Spark`, not the nvidia QAD checkpoint.)

## 2026-10-17 — deepseek isolation + fresh image pull
- The deepseek recipe `orcarouter-eugr-1m.yaml` used the SAME
  `eugr/spark-vllm-b12x:latest` tag as GLM => re-pulling :latest would silently move
  deepseek to the new build on its next launch.
- USER DECISION (recommended): PIN deepseek to the known-good digest
  `eugr/spark-vllm-b12x@sha256:d7213eea8cf9e5918e26ec75f46a505a9dc66a05fbae9c20ba508d3322c28d07`
  so it is isolated from :latest. Recipe backed up to orcarouter-eugr-1m.yaml.bak-20261017
  and patched (container: eugr/spark-vllm-b12x@sha256:d7213eea...). No deepseek running, so
  nothing live disturbed.
- Fresh `eugr/spark-vllm-b12x:latest` pull started on BOTH nodes as wfoster (rootless,
  NOT root — root podman uses a different, empty storage). Background pulls
  proc_c31faa748a8c (leo) / proc_a192ad3a569f (raph).
- Known issue: the earlier wasted pull was `sudo podman` (root) which is the WRONG
  storage; the serving containers are wfoster rootless.




