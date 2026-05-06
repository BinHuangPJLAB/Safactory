# DABStepEnv

Gymnasium wrapper for the DABStep benchmark, with automatic data download, multi-process sharding, and step-by-step rendering.

> This guide assumes AIEvoBox is installed and your current working directory is `AIEvoBox/`.

---

## Quick Start

### Step 1: Install the official evaluator (optional)

```bash
pip install git+https://huggingface.co/spaces/adyen/DABstep.git@main
```

> If it is not installed, the environment falls back to approximate scoring. Local debugging still works.

---

### Step 2: Configure environment parameters

Edit `env/dabstep/dabstep_config.yaml`:

```yaml
environments:
  - env_name: dabstepgym
    env_num: 1
    env_params:
      data_dir: "env/dabstep/data"
      artifacts_dir: "env/dabstep/artifacts"
      split: "dev"
      limit: 1
      shard_index: 0
      num_shards: 8
      max_steps: 10
      timeout: 60
```

**Data is downloaded automatically from Hugging Face on first run**, so no manual preparation is required. The downloaded directory layout is:

```text
env/dabstep/data/
├── context/          # CSV/JSON background data
└── tasks/
    ├── default_tasks.jsonl   # 450 tasks without answers, used for leaderboard submission
    └── dev_tasks.jsonl       # 10 tasks with answers, used for local scoring
```

---

### Step 3: Run

```bash
python launcher.py \
  --mode local \
  --manager-config manager/config.yaml \
  --env-config env/dabstep/dabstep_config.yaml \
  --llm-base-url <your-base-url> \
  --llm-api-key  <your-api-key> \
  --llm-model    <model-name> \
  --pool-size 1
```

---

### Step 4: Inspect results

After each task finishes, artifacts are written to `env/dabstep/artifacts/`:

```text
artifacts/
└── dabstep_20260309_193140_<task_id>/
    ├── env.log             # Full run log
    ├── trace.jsonl         # Per-step thought / code / output records
    ├── dev_metrics.json    # Scoring results, generated only for split=dev
    └── render_step_*.png   # Rendered screenshots, generated when render() is called
```

When using the `dev` split, open `dev_metrics.json` to view local scores.

---

## Split Notes

| split | Tasks | Answers | Purpose |
|-------|-------|---------|---------|
| `dev` | 10 | Yes | Local debugging and scoring |
| `default` | 450 | No | Official evaluation and leaderboard submission |

---

## Extension: Parallel Sharding

After a single shard works, change `split` to `default` and expand multiple shards in the config:

```yaml
environments:
  - env_name: dabstepgym
    env_num: 1
    env_params:
      data_dir: "env/dabstep/data"
      artifacts_dir: "env/dabstep/artifacts"
      split: "default"
      limit: 0
      shard_index: 0
      num_shards: 8

  - env_name: dabstepgym
    env_num: 1
    env_params:
      data_dir: "env/dabstep/data"
      artifacts_dir: "env/dabstep/artifacts"
      split: "default"
      limit: 0
      shard_index: 1
      num_shards: 8
  # shards 2-7 are configured the same way
```

After splitting `default` into 8 shards, each shard has about **56 tasks**. The `dev` split distribution is:

| shard_index | Tasks |
|-------------|-------|
| 0, 1 | 2 tasks |
| 2 ~ 7 | 1 task |

---

## Docker Deployment (environment tag TBD)

- Build the image:

```bash
docker build -f env/dabstep/Dockerfile -t dabstep:latest .
```

- To use a prebuilt image, replace the registry with your own image repository, for example:

```bash
docker pull <your-registry>/dabstep:latest
```

---

## Related Links

- [DABStep Hugging Face Dataset](https://huggingface.co/datasets/adyen/DABstep)
- [DABStep official evaluator](https://huggingface.co/spaces/adyen/DABstep)

## License

This adapter follows the AIEvoBox license. The DABStep dataset is licensed under CC-BY-4.0.
