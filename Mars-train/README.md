# MARS-SQL training

This directory contains the reinforcement-learning implementation and the
`mars-train.sh` entry point used for MARS-SQL training. The vendored
`SkyRL-OpenHands/` and `verl/` trees provide the agent environment and trainer
runtime; project-specific launch settings live in `mars-train.sh`.

## Setup

Follow [Install.md](./Install.md) to create the Python 3.12, CUDA 12.4, `uv`, and
Ray environment. The launch script can be called from any working directory and
uses the repository's top-level `data/` directory by default.

## Required configuration

| Variable | Purpose |
| :--- | :--- |
| `DB_PATH` | Root containing BIRD database directories such as `<db_id>/<db_id>.sqlite` |
| `CKPT_PATH` | Parent directory in which checkpoints are written |
| `WANDB_API_KEY` | W&B credential used by the configured training logger |

Optional variables include `DATA_DIR`, `CUDA_VISIBLE_DEVICES`, `PROJECT_NAME`,
and `EXPERIMENT_NAME`. Set `RESET_EXPERIMENT=1` only when the existing checkpoint
directory for the selected project and experiment should be removed. Set
`RESET_RAY=1` to stop an existing local Ray runtime before launch.

## Launch

```bash
export DB_PATH=/absolute/path/to/bird/databases
export CKPT_PATH=/absolute/path/to/checkpoints
export WANDB_API_KEY=your_wandb_api_key
bash Mars-train/mars-train.sh
```

The script validates its database directory and the required training Parquet
files before starting Ray/verl, so path errors fail early with a useful message.
