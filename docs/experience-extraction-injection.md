# Experience Extraction and Injection

Safactory can reuse historical trajectories as prompt-time experience. The workflow has two stages:

1. Extract reusable lessons from recorded trajectories.
2. Inject relevant lessons into future episodes.

## Files

| File or directory | Purpose |
|-------------------|---------|
| `exp_service/exp_service_config.example.yaml` | Example extraction-service config. |
| `core/exp/config.yaml` | Runtime prompt-injection config used by `launcher.py`. |
| `experiences/` | Suggested output directory for extracted lessons. |

## Extract Experiences

Run extraction after you have trajectory data in the Safactory database:

```bash
python3 -m exp_service run-once --config exp_service/exp_service_config.example.yaml
```

Copy the example config before using it for real experiments, then set the source database, output directory, model endpoint, and extraction options.

## Enable Prompt Injection

Update `core/exp/config.yaml`:

```yaml
enabled: true
dir: ./experiences
top_k: 3
mode: template
embedding_model: null
```

Supported modes:

| Mode | Behavior |
|------|----------|
| `template` | Selects high-ranked experience templates. |
| `ucb` | Uses a simple bandit strategy with usage feedback. |
| `contextual_ucb` | Uses task text and feedback history to choose experience. |

Set `enabled: false` to disable injection.

## Run With Injection

`launcher.py` reads `core/exp/config.yaml` by default. Use `--exp-config` to point at another file:

```bash
python3 launcher.py \
  --env-config env/osgym/os_config.yaml \
  --exp-config ./core/exp/config.yaml \
  --llm-base-url http://YOUR_LLM_HOST/v1 \
  --llm-api-key YOUR_API_KEY \
  --llm-model YOUR_MODEL
```

## Practical Notes

- Keep the extraction output directory and injection `dir` consistent.
- Start with `mode: template` for predictable behavior.
- Use separate experience directories for separate experiments to avoid mixing feedback statistics.
- Version experience libraries alongside the dataset or model run that produced them.
