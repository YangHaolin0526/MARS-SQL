#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

: "${DB_PATH:?Set DB_PATH to the root directory containing the BIRD databases.}"
: "${CKPT_PATH:?Set CKPT_PATH to the directory where checkpoints should be stored.}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY in your environment before training.}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"

PROJECT_NAME="${PROJECT_NAME:-MARS-SQL}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-Bird_train}"
EXPERIMENT_DIR="${CKPT_PATH%/}/${PROJECT_NAME}/${EXPERIMENT_NAME}"

if [[ ! -d "${DB_PATH}" ]]; then
    echo "ERROR: DB_PATH is not a directory: ${DB_PATH}" >&2
    exit 1
fi

for data_file in bird_train.parquet validation.parquet; do
    if [[ ! -f "${DATA_DIR}/${data_file}" ]]; then
        echo "ERROR: Missing training data: ${DATA_DIR}/${data_file}" >&2
        exit 1
    fi
done

mkdir -p -- "${CKPT_PATH}"
if [[ "${RESET_EXPERIMENT:-0}" == "1" ]]; then
    echo "INFO: Removing existing experiment directory: ${EXPERIMENT_DIR}"
    rm -rf -- "${EXPERIMENT_DIR}"
fi

KL_LOSS_COEF=0.001
ENTROPY_COEFF=0
KL_LOSS_TYPE=low_var_kl
N_AGENT=3
N_TURNS=5
TEMP=0.6
TOPP=0.95
USE_KL_LOSS=False
LR=1e-6
CLIP_LOW=0.2
CLIP_HIGH=0.2
GRAD_CLIP=0.5
BATCH_SIZE=128
TP_SIZE=4

if [[ "${RESET_RAY:-0}" == "1" ]]; then
    ray stop --force
fi

cd -- "${SCRIPT_DIR}"
PYTHONUNBUFFERED=1 uv run --extra sql --isolated --frozen -m verl.trainer.main_ppo \
    "data.train_files=${DATA_DIR}/bird_train.parquet" \
    "data.val_files=${DATA_DIR}/validation.parquet" \
    "data.train_batch_size=${BATCH_SIZE}" \
    data.dataloader_num_workers=0 \
    data.val_batch_size=1 \
    data.max_prompt_length=4096 \
    data.max_response_length=2572 \
    actor_rollout_ref.rollout.sql.max_prompt_length=4096 \
    actor_rollout_ref.rollout.sql.max_response_length=2572 \
    actor_rollout_ref.rollout.sql.max_start_length=2048 \
    actor_rollout_ref.rollout.sql.max_obs_length=512 \
    algorithm.adv_estimator=grpo \
    "actor_rollout_ref.model.path=${BASE_MODEL}" \
    "actor_rollout_ref.actor.use_kl_loss=${USE_KL_LOSS}" \
    "actor_rollout_ref.actor.clip_ratio_high=${CLIP_HIGH}" \
    "actor_rollout_ref.actor.clip_ratio_low=${CLIP_LOW}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.model.use_remove_padding=True \
    "actor_rollout_ref.actor.optim.lr=${LR}" \
    "actor_rollout_ref.actor.ppo_mini_batch_size=${BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    "actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}" \
    "actor_rollout_ref.actor.kl_loss_type=${KL_LOSS_TYPE}" \
    "actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=64 \
    "actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE}" \
    "actor_rollout_ref.rollout.n_trajectories=${N_AGENT}" \
    "actor_rollout_ref.rollout.max_iterations=${N_TURNS}" \
    actor_rollout_ref.rollout.name=async \
    actor_rollout_ref.rollout.enable_memory_saver=True \
    actor_rollout_ref.rollout.task_type=sql \
    reward_model.reward_manager=sql \
    "actor_rollout_ref.rollout.sql.db_path=${DB_PATH}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=64 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    "actor_rollout_ref.rollout.sampling_params.temperature=${TEMP}" \
    "actor_rollout_ref.rollout.sampling_params.top_p=${TOPP}" \
    actor_rollout_ref.actor.masking=true \
    "trainer.logger=['wandb']" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=15 \
    trainer.test_freq=-1 \
    "trainer.project_name=${PROJECT_NAME}" \
    "trainer.experiment_name=${EXPERIMENT_NAME}" \
    trainer.total_epochs=20 \
    "trainer.default_local_dir=${EXPERIMENT_DIR}" \
    trainer.max_actor_ckpt_to_keep=10 \
    trainer.max_critic_ckpt_to_keep=10 \
    +trainer.load_checkpoint=null \
    "actor_rollout_ref.actor.grad_clip=${GRAD_CLIP}" \
    2>&1
