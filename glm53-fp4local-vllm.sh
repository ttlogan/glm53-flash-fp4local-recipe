#!/usr/bin/env bash
# GLM-5.3-Flash-FP4-Local (nvidia / ModelOpt QAD NVFP4), vLLM B12X TP=2 on
# 2x ASUS GX10 (GB10 / sm_121a). Native MTP spec-decode (NO external drafter).
# Model is resolved from wfoster's HF cache (snapshot dir).
#
# NOTE (staged, not yet validated): the --load-format / spec-decode / fabric env
# below are the nvidia card's 2x-DGX-Spark settings, reconciled with this
# cluster's proven b12x fabric (NCCL_NET=Socket, NCCL_IB_DISABLE=1, the same
# config the deepseek b12x lane runs). Validate at the gate before going live;
# the card also ships a RoCE-twin variant (NCCL_IB_DISABLE=0, VLLM_ENABLE_ROCE_ALLREDUCE=1).
set -euo pipefail

NODE_RANK="${1:?usage: glm53-fp4local-vllm.sh <0|1>}"
[[ "$NODE_RANK" == "0" || "$NODE_RANK" == "1" ]] || { echo "rank must be 0 or 1" >&2; exit 2; }

IMAGE="docker.io/eugr/spark-vllm-b12x:latest"
NAME="vllm_glm53"
HF_CACHE="/home/wfoster/.cache/huggingface"
HF_REPO="models--nvidia--GLM-5.3-Flash-FP4-Local"
REVISION="76348ddb2be2e35e39ecfc00e75dc07703f4f3d5"
# cluster fabric: leo=.11 (head/rank0), raph=.12 (worker/rank1); master = leo
HEAD_IP="192.168.177.11"
MPORT="29521"
PORT="8000"

case "$NODE_RANK" in
  0) HOST_IP=192.168.177.11; HEADLESS="" ;;
  1) HOST_IP=192.168.177.12; HEADLESS="--headless" ;;
esac

# Resolve the model snapshot dir inside the container (HF cache is bind-mounted).
CMODEL_DIR="$(ls -d "/root/.cache/huggingface/hub/${HF_REPO}/snapshots/"* 2>/dev/null | head -1 || true)"
[[ -n "$CMODEL_DIR" && -f "${CMODEL_DIR}/config.json" ]] || {
  echo "nvidia GLM-5.3-Flash-FP4-Local not in HF cache: ${HF_CACHE}/hub/${HF_REPO} (still downloading?)" >&2
  exit 4
}

podman rm -f "$NAME" 2>/dev/null || true

podman run -d \
  --name "$NAME" --restart no \
  --device nvidia.com/gpu=all \
  --network host --ipc host \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
  --device /dev/infiniband:/dev/infiniband \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  -e VLLM_HOST_IP="$HOST_IP" \
  -e HF_HOME=/root/.cache/huggingface -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e NCCL_NET=Socket -e NCCL_IB_DISABLE=1 \
  -e NCCL_IB_HCA=rocep1s0f0 -e NCCL_IB_GID_INDEX=3 -e NCCL_IB_ROCE_VERSION_NUM=2 \
  -e NCCL_IB_ADDR_FAMILY=AF_INET -e NCCL_IB_ADDR_RANGE=192.168.177.0/24 \
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e GLOO_SOCKET_IFNAME=enp1s0f0np0 \
  -e TP_SOCKET_IFNAME=enp1s0f0np0 -e MN_IF_NAME=enp1s0f0np0 \
  -e NCCL_NVLS_ENABLE=0 -e NCCL_CROSS_NIC=0 -e NCCL_IB_MERGE_NICS=0 \
  -e CUTE_DSL_ARCH=sm_121a \
  -e SAFETENSORS_FAST_GPU=1 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e B12X_POLICY_MODE=auto \
  -e VLLM_B12X_MOE_FP4_FORCE_A16=1 \
  -e B12X_W4A16_A4_PREFILL=1 \
  -e INSTANTTENSOR_BACKEND=BUFFERED -e INSTANTTENSOR_BUFFER_SIZE=67108864 \
  -e INSTANTTENSOR_CHUNK_SIZE=8388608 -e INSTANTTENSOR_CONCURRENCY=1 -e INSTANTTENSOR_IO_DEPTH=3 \
  "$IMAGE" \
    "${CMODEL_DIR}" \
    --served-model-name dgx_hobo_default \
    --host 0.0.0.0 --port "$PORT" \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --quantization modelopt_mixed \
    --attention-backend B12X \
    --moe-backend b12x \
    --linear-backend b12x \
    --load-format instanttensor \
    --mamba-cache-mode align \
    --kv-cache-dtype fp8 \
    --block-size 256 \
    --gpu-memory-utilization 0.87 \
    --max-model-len 1048576 \
    --max-num-seqs 4 \
    --max-num-batched-tokens 4096 \
    --decode-context-parallel-size 2 \
    --enable-prefix-caching --enable-chunked-prefill \
    --no-enable-flashinfer-autotune \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"b12x","attention_backend":"B12X"}' \
    --tool-call-parser glm47 --enable-auto-tool-choice \
    --reasoning-parser glm45 \
    --default-chat-template-kwargs '{"reasoning_effort":"max","clear_thinking":false}' \
    --chat-template "${CMODEL_DIR}/chat_template.jinja" \
    --distributed-executor-backend mp \
    --nnodes 2 --node-rank "$NODE_RANK" \
    --master-addr "$HEAD_IP" --master-port "$MPORT" \
    $HEADLESS

echo "launched $NAME rank=$NODE_RANK host=$HOST_IP"
sleep 2
podman ps --format '{{.Names}} {{.Status}}' | grep "$NAME" || { echo "$NAME exited; inspect: podman logs $NAME" >&2; exit 1; }
