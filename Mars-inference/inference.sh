#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

: "${DB_PATH:?Set DB_PATH to the root directory containing the benchmark databases.}"

DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/bird_test.parquet}"
SAVE_PATH="${SAVE_PATH:-${SCRIPT_DIR}/step80_bird_@16_turn5_test_result.parquet}"
MODEL_PATH="${MODEL_PATH:-Yanghl0526/Qwen-SQL-7B-bird_5turns_80step}"
STABLE_WORKDIR="${STABLE_WORKDIR:-${SCRIPT_DIR}/outputs/hydra}"

if [[ ! -f "${DATA_PATH}" ]]; then
    echo "ERROR: DATA_PATH is not a file: ${DATA_PATH}" >&2
    exit 1
fi
if [[ ! -d "${DB_PATH}" ]]; then
    echo "ERROR: DB_PATH is not a directory: ${DB_PATH}" >&2
    exit 1
fi

mkdir -p -- "$(dirname -- "${SAVE_PATH}")" "${STABLE_WORKDIR}"
cd -- "${SCRIPT_DIR}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WORLD_SIZE="${WORLD_SIZE:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export RANK="${RANK:-0}"
export LOCAL_RANK="${LOCAL_RANK:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"

python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=1 \
    "data.path=${DATA_PATH}" \
    data.prompt_key=prompt \
    data.n_samples=1 \
    data.batch_size=8 \
    "data.output_path=${SAVE_PATH}" \
    "+data.base_db_path=${DB_PATH}" \
    "model.path=${MODEL_PATH}" \
    +model.trust_remote_code=True \
    rollout.name=async \
    rollout.temperature=0.8 \
    rollout.top_k=50 \
    rollout.top_p=0.7 \
    rollout.prompt_length=3096 \
    rollout.response_length=5096 \
    rollout.tensor_model_parallel_size=1 \
    rollout.gpu_memory_utilization=0.8 \
    +rollout.task_type=sql \
    +rollout.port=30000 \
    +rollout.max_iterations=5 \
    +rollout.sql.max_start_length=3048 \
    +rollout.sql.max_prompt_length=3096 \
    +rollout.sql.max_response_length=5096 \
    +rollout.sql.max_obs_length=1024 \
    "+rollout.sql.db_path=${DB_PATH}" \
    +rollout.n_trajectories=16 \
    +rollout.sampling_params.max_new_tokens=1024 \
    "hydra.run.dir=${STABLE_WORKDIR}"
