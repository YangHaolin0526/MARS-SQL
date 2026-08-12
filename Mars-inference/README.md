# MARS-SQL inference and evaluation

This directory contains the MARS-SQL generation entry point, execution-based
evaluation, and three candidate-selection strategies.

## Environment

```bash
conda create -n mars-infer python=3.10 -y
conda activate mars-infer
pip install -r Mars-inference/requirements.txt
```

## Generate candidates

`inference.sh` can be called from any working directory. `DB_PATH` is required;
the other paths have repository-relative defaults.

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `DB_PATH` | required | Root containing `<db_id>/<db_id>.sqlite` databases |
| `DATA_PATH` | `data/bird_test.parquet` | Prepared evaluation prompts |
| `SAVE_PATH` | `Mars-inference/step80_bird_@16_turn5_test_result.parquet` | Generated trajectories |
| `MODEL_PATH` | `Yanghl0526/Qwen-SQL-7B-bird_5turns_80step` | Local or Hugging Face model identifier |
| `STABLE_WORKDIR` | `Mars-inference/outputs/hydra` | Hydra run metadata directory |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU selection |

```bash
export DB_PATH=/absolute/path/to/bird/databases
bash Mars-inference/inference.sh
```

## Evaluate

```bash
python Mars-inference/evaluate_sql.py \
  --input_file Mars-inference/step80_bird_@16_turn5_test_result.parquet \
  --db_path /absolute/path/to/bird/databases \
  --num_cpus 16
```

The evaluator opens SQLite databases in read-only mode and writes a TSV decision
log plus a text score summary. Tune `--timeout` and `--num_cpus` for the host.

## Candidate selection

| Script | Strategy |
| :--- | :--- |
| `select_self_consistency.py` | Select by agreement among execution results |
| `select_with_genrm.py` | Score candidates with a generative reward model |
| `select_best_sql_api.py` | Select with an OpenAI-compatible API |

Run any script with `--help` for its complete input, database, model/API, and
output options. Pass API credentials at runtime; do not commit them to the
repository.
