# Geo3K-VL

Geo3K-VL is the geometry visual-language reasoning environment in this repository. Its runtime registry name is `geo3k_vl_test`. Each task usually includes:

- `problem` or `question`: problem text
- `answer`: reference answer
- `images`: list of image URLs

The environment supports multi-turn interaction and also allows the model to call the built-in scoring tool before producing a final answer.

## Configuration Files

Start from `env/geo3k_vl_test/geo3k_vl_test_env_configs.yaml`:

| Field | Description |
|-------|-------------|
| `env_name: geo3k_vl_test` | Environment registry name |
| `dataset` | Parquet dataset path |
| `env_params.config_path` | Optional runtime config file path |

Runtime configuration is loaded by default from `env/geo3k_vl_test/geo3k_vl_test_env_runtime.yaml` and mainly includes:

| Parameter | Description |
|-----------|-------------|
| `max_turns` | Maximum interaction turns |
| `max_images` | Maximum number of images injected into the prompt |
| `echo_images_on_feedback` | Whether tool feedback repeats the images |

## Run Example

```bash
python launcher.py \
  --mode local \
  --env-config env/geo3k_vl_test/geo3k_vl_test_env_configs.yaml \
  --llm-base-url http://YOUR_LLM_HOST/v1 \
  --llm-api-key YOUR_LLM_API_KEY \
  --llm-model YOUR_MODEL_NAME \
  --pool-size 1
```

## Notes

- Public documentation uses `Geo3K-VL` for this environment, while the repository registry name remains `geo3k_vl_test`.
- Before running, replace the dataset path with a locally accessible parquet file path.
