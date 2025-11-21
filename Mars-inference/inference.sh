set -x
export TMPDIR=''
export RAY_TMPDIR=''

export HF_HUB_OFFLINE=1

data_path=/data/bird_test.parquet

save_path=step80_bird_@16_turn5_test_result.parquet
# model_path=Qwen/Qwen2.5-Coder-7B-Instruct
model_path=Yanghl0526/Qwen-SQL-7B-bird_5turns_80step

# sleep 5
export CUDA_VISIBLE_DEVICES=6

export WORLD_SIZE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export RANK=0
export LOCAL_RANK=0
export NCCL_P2P_DISABLE="1"
STABLE_WORKDIR=""
mkdir -p $STABLE_WORKDIR

python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=1 \
    data.path=$data_path \
    data.prompt_key=prompt \
    data.n_samples=1 \
    data.batch_size=8 \
    data.output_path=$save_path \
    +data.base_db_path=bird_databse \
    model.path=$model_path \
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
    +rollout.sql.db_path=bird_database \
    +rollout.n_trajectories=16 \
    +rollout.sampling_params.max_new_tokens=1024 \
    hydra.run.dir=$STABLE_WORKDIR
    # +rollout.enable_memory_saver=True \
    # +trainer.hybrid_engine=True
