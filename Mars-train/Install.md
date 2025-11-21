## Pre-requisites


The main prerequisites are: 
- [CUDA Toolkit 12.4](https://developer.nvidia.com/cuda-12-4-0-download-archive) (versions greater than 12.4 might also work)
- `build-essential`: This is needed for `torch-memory-saver`
- [`uv`](https://docs.astral.sh/uv/getting-started/installation): We use the `uv` + `ray` integration to easily manage dependencies in multi-node training.
- `python` 3.12
- `ray` 2.43.0


Once installed, configure ray to use `uv` with 

```
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
```


## Installation & Verification

Ensure you are in the project root directory where pyproject.toml and uv.lock are located:

```bash
cd MARS-SQL/Mars-train
```

We rely on `uv` to manage the environment and lock dependencies strictly. You can verify your setup by performing a "dry run" which initializes Ray within a `uv`-managed environment.

### 1. Base Installation Dry Run

Run the following command to verify that the base environment (Ray, Torch, etc.) can be installed and initialized correctly:

```bash
uv run --isolated --frozen python -c 'import ray; ray.init(); print("Success!")'
```

### 2. SQL Support Dry Run

If you require SQL functionalities (e.g., text-to-sql tasks), we have defined an optional dependency group named `sql`. Run the following command to verify the environment with SQL extras:

```bash
uv run --isolated --extra sql --frozen python -c 'import ray; ray.init(); print("Success!")'
```

> [!NOTE]
> The `--frozen` flag ensures that the exact versions specified in `uv.lock` are used. The `--isolated` flag ensures the run does not interfere with your global Python packages.