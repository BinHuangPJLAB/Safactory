# Reinforcement Learning Training

Safactory integrates with [Slime](https://github.com/THUDM/slime) through a Buffer Server. Safactory collects rollout trajectories, the Buffer Server batches completed trajectories, and Slime pulls batches for GRPO-style training.

## Architecture

```text
Safactory launcher.py
        |
        | rollout trajectories
        v
Buffer Server (rl/buffer_server.py)
        |
        | training batches
        v
Slime / GRPO training process
```

The rollout process and training process can run on different machines if the ports are reachable.

## Prerequisites

- Safactory dependencies installed with `pip install -r requirements.txt`.
- Slime installed for the training process.
- At least one Safactory environment config that can run successfully.
- Network access between the Buffer Server, LLM proxy, and Slime process when distributed.

Install Slime using either the upstream build script or the official Docker workflow documented in the Slime repository.

## Configure

Export the RL variables in your shell or keep them in a local `.env` file that your launch wrapper sources. This checkout does not include a committed `rl/.env.example`.

The current Buffer Server reads the legacy `AIEVOBOX_*` variable prefix, so use those names unless the code is updated.

## Start Training

Terminal 1, Slime process:

```bash
cd rl

# Text-only models
./run_slime_generator.sh

# Vision-language models
./run_slime_generator_vl.sh
```

Terminal 2, Buffer Server and Safactory rollout runner:

```bash
cd rl
./run_buffer_server.sh
```

## Key Variables

Environment selection:

| Variable | Description |
|----------|-------------|
| `AIEVOBOX_ROOT` | Safactory repository root used by the Buffer Server to find `launcher.py`. |
| `AIEVOBOX_ENV_CONFIG` | Path to one environment YAML for focused runs. |
| `AIEVOBOX_ENV_ROOT` | Directory scanned for multiple environment YAML files. |
| `AIEVOBOX_ENV_TRANSPORT` | Launcher environment transport, default `http`. |
| `AIEVOBOX_DB_URL` | SQLite URI for RL rollout storage. |
| `AIEVOBOX_POOL_SIZE` | Runner pool size. |
| `AIEVOBOX_MAX_STEPS` | Maximum episode steps. |
| `AIEVOBOX_MESSAGE_CUT` | Message history retention. |

Training:

| Variable | Description |
|----------|-------------|
| `RL_GROUP_SIZE` | Samples per prompt, mapped to Slime sampling settings. |
| `RL_EPOCH` | Number of rollout epochs. |
| `RL_OFF_BY_N` | Maximum allowed model weight-version lag. `0` means strictly on-policy. |
| `SLIME_GLOBAL_BATCH_SIZE` | Global training batch size. |

Services:

| Service | Default port | Override |
|---------|--------------|----------|
| Buffer Server | `18889` | `BUFFER_SERVER_PORT` |
| LLM Proxy | `18890` | `LLM_PROXY_PORT` |

## Notes

- The Buffer Server machine needs Safactory dependencies but does not need Slime.
- The Slime training machine needs Slime and access to the Buffer Server.
- Start with one environment and small `RL_GROUP_SIZE` before scaling up.
- Keep model endpoint, dataset paths, and output directories explicit in `.env` for reproducibility.
