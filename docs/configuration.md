# Configuration

Safactory configuration has three layers:

1. CLI flags passed to `launcher.py`.
2. The manager config at `manager/config.yaml`.
3. Environment YAML files under `env/` or passed with `--env-config`.

CLI flags take precedence over values loaded from YAML.

## Essential CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `local` | Execution mode: `local` or `remote`. |
| `--env-transport` | `http` | Environment transport: `http` or `inproc`. |
| `--env-config` | `None` | Path to one environment YAML. Incompatible with `--env-root`. |
| `--env-root` | `env` | Directory scanned recursively for environment YAML files when `--env-config` is not set. |
| `--pool-size` | `0` | Override YAML pool size. `0` keeps the YAML value. |
| `--llm-base-url` | project default endpoint | OpenAI-compatible model endpoint. Replace this for real runs. |
| `--llm-api-key` | `EMPTY` | API key. Use `EMPTY` for endpoints that do not require one. |
| `--llm-model` | project default model | Model identifier sent to the endpoint. |
| `--max-steps` | `1000` | Maximum agent steps per episode. |
| `--db-path` | `sqlite://android_envs.db` | SQLite database URI. |
| `--log-dir` | `logs` | Run log directory. |

Example:

```bash
python launcher.py \
  --mode local \
  --env-transport http \
  --env-config env/osgym/os_config.yaml \
  --llm-base-url http://YOUR_LLM_HOST/v1 \
  --llm-api-key YOUR_API_KEY \
  --llm-model YOUR_MODEL \
  --pool-size 1
```

## Full CLI Reference

| Category | Flag | Default | Description |
|----------|------|---------|-------------|
| Job | `--job-id` | auto-generated when empty | Identifier recorded with environment configs and session rows. |
| Config | `--manager-config` | `./manager/config.yaml` | Manager YAML path. |
| Config | `--exp-config` | `./core/exp/config.yaml` | Experience injection YAML path. |
| Storage | `--storage-type` | `sqlite` | Storage backend: `sqlite` or `cloud`. |
| Storage | `--warmup-count` | `100` | Number of environment configs preloaded into the manager. |
| Storage | `--save-batch-size` | `100` | Batch size for storing environment configs. |
| Storage | `--disable-buffer` | buffer enabled | Disable buffered record writes. |
| Storage | `--buffer-size` | `100` | Buffered record capacity. |
| Storage | `--flush-interval` | `5.0` | Buffered write flush interval in seconds. |
| Storage | `--rebuild-table` / `--no-rebuild-table` | `false` | Delete and recreate SQLite tables before loading configs. |
| Pool | `--multiplier` | `1.2` | Pre-warm `ceil(multiplier * pool_size)` actors. |
| Local service | `--start-local-upstream` / `--no-start-local-upstream` | `None` | Explicitly control local FastAPI service startup. |
| Local service | `--local-upstream-app` | `env.app:app` | ASGI app used for local HTTP mode. |
| Local service | `--local-upstream-host` | `0.0.0.0` | Local service bind host. |
| Local service | `--local-upstream-port` | `36663` | Local service port. |
| Local service | `--local-upstream-url` | `http://127.0.0.1:36663` | URL used by the runner to reach the local service. |
| Local service | `--wait-timeout` | `60.0` | Seconds to wait for local service readiness. |
| Interactor | `--message-cut` | `-1` | Number of recent messages kept in context. `<= 0` keeps all. |
| Interactor | `--env-http-timeout-s` | `300.0` | Environment HTTP request timeout. |
| Interactor | `--http-retries` | `2` | Retry count for failed environment calls. |
| LLM | `--llm-temperature` | `0.3` | Sampling temperature. |
| RL | `--rl-use-session-suffix-url` | `false` | Use session-routed LLM proxy URLs. |
| RL | `--rl-group-size` | `0` | Override `env_num` for repeated samples per prompt. |
| RL | `--rl-epoch` | `1` | Duplicate environment configs for multiple RL epochs. |
| Logging | `--run-name` | empty | Optional prefix for the run log directory. |
| Logging | `--console-log-level` | `INFO` | Console log level. |
| Logging | `--file-log-level` | `DEBUG` | File log level. |
| Logging | `--log-max-bytes` | `52428800` | Legacy rotation size. Ignored in run-directory log mode. |
| Logging | `--log-backup-count` | `20` | Recent run log directories to keep. `0` keeps all. |
| Logging | `--debug-log` | `false` | Enable debug storage logs on the console. |

## Manager Config

`manager/config.yaml` controls database defaults, pool size, remote cluster settings, environment launch commands, and RayJob credentials. Treat checked-in values as examples and replace private paths, image names, and credentials before use.

Minimal local-style example:

```yaml
mode: local
pool_size: 1

database:
  driver: sqlite
  sqlite_path: android_envs.db

cluster:
  http:
    port: 36663
    timeout_s: 100
    concurrency: 200
    startup_concurrency: 16
```

Remote mode adds per-environment cluster launch settings:

```yaml
cluster:
  env_types:
    os_gym:
      quotagroup: "your-quota"
      entrypoint: "bash /path/to/start_os_gym.sh"
      resources:
        head:
          cpu: 18
          gpu: 0
          memory: 72Gi
          privileged: true
      limit: 12

rayjob:
  domain: "https://your-rayjob-platform.example"
  tenant: "your-tenant"
  access_key: "replace-me"
  secret_key: "replace-me"
  project: "your-project"
```

## Environment YAML

Each environment YAML defines one or more runnable environment groups.

```yaml
environments:
  - env_name: os_gym
    env_image: your_optional_runtime_image
    env_num: 1
    dataset: path/to/cases.jsonl
    dataset_load_mode: eager
    env_params:
      max_steps: 30
      prompt_format: kimi
```

| Field | Required | Description |
|-------|----------|-------------|
| `env_name` | yes | Registry name resolved by `env/env_factory.py`. |
| `env_image` | no | Runtime image used by remote launchers. |
| `env_num` | no | Number of parallel instances for this environment group. |
| `dataset` | no | JSON, JSONL, YAML, text, or parquet path consumed by the data loader. |
| `dataset_load_mode` | no | Defaults to `eager`; some parquet-based environments use `parquet_row_ref`. |
| `env_params` | no | Keyword arguments passed into the environment constructor. |

Supported registry names include `android_gym`, `os_gym`, `mc`, `mc_gym`, `embodied_alfred`, `emb`, `qa_gym`, `dabstepgym`, `discoveryworld`, `deepeyes_env`, `geo3k_vl_test`, `robotrustbench`, `math500_text`, and `math500`.

## Data Paths

Relative dataset paths are resolved from the YAML file directory. For example, `dataset: cases.jsonl` in `env/androidgym/android_env.yaml` resolves to `env/androidgym/cases.jsonl`.

Use absolute paths for shared storage, Docker-mounted volumes, or externally downloaded datasets.
