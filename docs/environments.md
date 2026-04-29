# Supported Environments

![Safactory architecture](../fig/envs.png)

Safactory ships multiple environment adapters behind one launcher and one `BaseEnv` interface. Some run directly on the host, while VM, emulator, simulator, or Docker-backed environments need extra setup.

## Overview

| Domain | Registry name | Public name | Notes | Config / guide |
|--------|---------------|-------------|-------|----------------|
| Desktop | `os_gym` | OSWorld / RiOSWorld | Ubuntu desktop automation with screenshots and `pyautogui`. | [guide](../env/osgym/README.md) |
| Mobile | `android_gym` | AndroidGym | Android emulator interaction through ADB. | [guide](../env/androidgym/README_EN.md) |
| Game | `mc`, `mc_gym` | Minecraft / MineStudio | Minecraft tasks through MineStudio / Malmo. | [install](../env/mc/INSTALL.md) |
| Embodied | `robotrustbench` | RoboTrustBench | Habitat-based safety and robustness tasks. | [guide](../env/robotrustbench/README.md) |
| Embodied | `embodied_alfred`, `emb` | Embodied ALFRED | Household embodied tasks through EmbodiedBench / ALFRED. | [guide](../env/embodiedgym/README.md) |
| QA | `qa_gym` | QAGym | Prompt-attack and QA robustness environment. | [guide](../env/qagym/README.md) |
| Data processing | `dabstepgym` | DABStep | Code-execution data wrangling tasks. | [guide](../env/dabstep/README.md) |
| Scientific discovery | `discoveryworld` | DiscoveryWorld | Text and optional vision science tasks. | [guide](../env/dwgym/README.md) |
| Multimodal reasoning | `deepeyes_env` | DeepEyes | Multi-turn visual tool-use tasks. | [guide](../env/deepeyes/README.md) |
| Geometry VL | `geo3k_vl_test` | Geo3K-VL | Geometry-focused visual reasoning tasks. | [guide](../env/geo3k_vl_test/README.md) |
| Math | `math500_text`, `math500` | Math500 text | Text-only math environment. | [config](../env/math500_text/math500_text_env_configs.yaml) |

## Environment Notes

### OS (`os_gym`)

Wraps OSWorld / RiOSWorld style desktop tasks. The agent observes a Linux desktop and acts through `pyautogui`.

Requirements:

- Docker or a compatible VM runtime.
- Privileged execution for QEMU / KVM when using VM-backed tasks.
- Ubuntu VM image, either downloaded automatically or provided through `vm_path`.
- OS task dataset in JSONL format.

### Android (`android_gym`)

Drives Android Emulator instances through ADB. The adapter supports parallel emulator mode and single snapshot mode.

Requirements:

- `adb` and Android Emulator available on the host or inside the runtime image.
- A compatible AVD, defaulting to `nexus_safe` unless overridden.
- Android task JSONL file, usually `env/androidgym/cases.jsonl`.

### Minecraft (`mc`)

Runs Minecraft tasks through MineStudio and Malmo-compatible tooling.

Requirements:

- Java 8 for Malmo compatibility.
- Xvfb for headless display.
- MineStudio dependencies from `env/mc/requirements.txt`.
- Optional CUDA support for GPU-accelerated components.

### RoboTrustBench (`robotrustbench`)

Runs embodied safety and robustness variants: `safety`, `robust`, and `robustd`.

Requirements:

- Habitat and simulator dependencies.
- Prepared task resources and dataset files for the selected variant.
- A containerized runtime is recommended.

### Embodied ALFRED (`embodied_alfred`, `emb`)

Adapts EmbodiedBench / ALFRED tasks into Safactory.

Requirements:

- EmbodiedBench installed.
- EB-ALFRED dataset.
- AI2-THOR resources.
- Xvfb and required system fonts for headless rendering.

### QAGym (`qa_gym`)

Models QA robustness and prompt-attack evaluation as an environment.

Requirements:

- QAGym dependencies from the repo root.
- OpenAI-compatible endpoints for the agent and optional judge or attack models.
- `env/qagym/qa_cases.jsonl` or a replacement dataset.

### DABStep (`dabstepgym`)

Runs data-wrangling tasks where agents write and execute Python code.

Requirements:

- DABStep dependencies.
- Optional official evaluation library from the DABStep Hugging Face Space.
- Dataset downloaded automatically or placed under `env/dabstep/data`.

### DiscoveryWorld (`discoveryworld`)

Provides scientific discovery tasks with text observations and optional visual frames.

Requirements:

- DiscoveryWorld installed under `env/dwgym/discoveryworld`.
- Scenario, difficulty, seed, and vision settings configured in YAML.

### DeepEyes (`deepeyes_env`)

Runs multimodal visual tool-use tasks with optional crop and rotate tools.

Requirements:

- Parquet dataset path.
- Optional judge model endpoint.
- Runtime config such as `env/deepeyes/deepeyes_env_runtime.yaml`.

### Geo3K-VL (`geo3k_vl_test`)

Runs geometry visual-language tasks with image inputs and reference answers.

Requirements:

- Parquet dataset path.
- Runtime config such as `env/geo3k_vl_test/geo3k_vl_test_env_runtime.yaml`.

## Running One Environment

```bash
python launcher.py \
  --mode local \
  --env-config env/osgym/os_config.yaml \
  --llm-base-url http://YOUR_LLM_HOST/v1 \
  --llm-api-key YOUR_API_KEY \
  --llm-model YOUR_MODEL \
  --pool-size 1
```

Use `--env-root env` when you want to load multiple YAML files recursively.
