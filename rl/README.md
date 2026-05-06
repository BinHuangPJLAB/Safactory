# RL Training

## How to Run

### 1. Configure environment variables

```bash
cp .env.example .env
# Edit .env and fill in the real configuration
```

### 2. Start Slime training (Terminal 1)

Slime must be installed according to the official guide before starting training.

Option 1: [shell script](https://github.com/THUDM/slime/blob/main/build_conda.sh) for install from scratch

Option 2: official Slime [Docker workflow](https://github.com/THUDM/slime/blob/main/docs/en/get_started/quick_start.md#pull-and-start-docker-container)

Start the training process:

```bash
cd /root/AIEvoBox/rl

# Text model
./run_slime_generator.sh

# Math500 text model (OPD + sglang teacher)
./examples/math500/run_slime_generator_opd_sglang.sh

# VL model
./run_slime_generator_vl.sh
```

### 3. Start Buffer Server (Terminal 2)

Only the main dependencies of this project are required. See the root README; Slime does not need to be installed here.

```bash
cd /root/AIEvoBox/rl
./run_buffer_server.sh
```

After startup, Buffer Server automatically starts the AIEvoBox Runner and begins collecting rollout data.

Terminal 1 and Terminal 2 can run on different machines as long as they can communicate.

## Key Configuration

### Environment configuration (choose one)

- `AIEVOBOX_ENV_CONFIG` specifies a single YAML file, suitable for debugging or running one environment.
- `AIEVOBOX_ENV_ROOT` specifies the environment root directory and automatically loads YAML files under all subdirectories.

### Training parameters

| Variable | Description |
|----------|-------------|
| `RL_GROUP_SIZE` | Samples per prompt, corresponding to Slime's `n_samples_per_prompt` |
| `RL_EPOCH` | Training epochs |
| `RL_OFF_BY_N` | Maximum allowed weight-version lag; 0 means only current-version data |
| `SLIME_GLOBAL_BATCH_SIZE` | Global batch size |

### Service ports

| Service | Default port | Environment variable |
|---------|--------------|----------------------|
| Buffer Server | 18889 | `BUFFER_SERVER_PORT` |
| LLM Proxy | 18890 | `LLM_PROXY_PORT` |

See `.env.example` for the full configuration. For more parameters in the Slime training scripts, refer to the Slime documentation.
