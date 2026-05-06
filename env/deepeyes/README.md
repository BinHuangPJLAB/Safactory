# DeepEyes

DeepEyes is a multi-turn visual tool-use environment in AIEvoBox. Each task usually includes:

- `prompt`: initial message
- `images`: image paths, URLs, or data URLs
- `reward_model.ground_truth`: reference answer
- `extra_info.question`: original question text

The environment parses tool calls from model outputs, supports visual operations such as cropping, and computes rewards at the end using the judge configuration.

## Configuration Files

Start from `env/deepeyes/deepeyes_env.yaml`. Common fields:

| Field | Description |
|-------|-------------|
| `env_name: deepeyes_env` | Environment registry name |
| `dataset` | Parquet dataset path |
| `dataset_load_mode: parquet_row_ref` | Lazy loading by parquet row reference |
| `env_params.config_path` | Runtime config file, defaulting to `deepeyes_env_runtime.yaml` |

Other optional example configs:

- `env/deepeyes/deepeyes_visual_08_env.yaml`
- `env/deepeyes/deepeyes_visual_012_env.yaml`
- `env/deepeyes/deepeyes_thinklite_env.yaml`
- `env/deepeyes/deepeyes_mixed_v1v8_thinklite_env.yaml`

## Runtime Configuration

`env/deepeyes/deepeyes_env_runtime.yaml` mainly controls:

| Parameter | Description |
|-----------|-------------|
| `max_turns` | Maximum turns allowed for one sample |
| `allow_rotate_tool` | Whether rotation tools are allowed |
| `http_timeout_s` | Image / network request timeout |
| `judge_base_url` / `judge_model` / `judge_api_key` | Judge model configuration |
| `cleanup_temp_dir_on_close` | Whether to delete the temporary crop directory on close |
| `crop_root` | Directory for temporary crop results |

## Run Example

```bash
python launcher.py \
  --mode local \
  --env-config env/deepeyes/deepeyes_env.yaml \
  --llm-base-url http://YOUR_LLM_HOST/v1 \
  --llm-api-key YOUR_LLM_API_KEY \
  --llm-model YOUR_MODEL_NAME \
  --pool-size 1
```

## Notes

- Before running, replace the dataset path, judge model address, and output directory with locally available values.
- If no extra judge is needed, leave `judge_model` empty in the runtime config or replace it with your own scoring endpoint.
