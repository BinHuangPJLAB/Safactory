# DiscoveryWorldEnv

This adapter integrates the [DiscoveryWorld](https://github.com/allenai/discoveryworld) benchmark into AIEvoBox, supporting text and vision modes, multiple scenarios, and multiple difficulty levels.

> This guide assumes AIEvoBox is installed and your current working directory is `AIEvoBox/`.

---

## Quick Start

### Step 1: Install DiscoveryWorld

```bash
cd env/dwgym
git clone https://github.com/allenai/discoveryworld.git
cd discoveryworld
pip install -e .
```

---

### Step 2: Configure environment parameters

Edit `env/dwgym/dw_config.yaml`:

```yaml
environments:
  - env_name: discoveryworld
    env_num: 1
    env_params:
      scenario_name: "Proteomics"
      difficulty: "Easy"
      seed: 0
      max_steps: 100
      use_vision: false
      narrate_actions: true
      max_recent_actions: 5
```

Full parameter reference:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `scenario_name` | str | first scenario | Scenario name; see the scenario list below |
| `difficulty` | str | `"Normal"` | Difficulty: `"Easy"` / `"Normal"` / `"Challenge"` |
| `seed` | int | `0` | Random seed |
| `max_steps` | int | `300` | Maximum interaction steps per episode |
| `use_vision` | bool | `false` | Whether to enable 2D visual observations |
| `capture_frames` | bool | `false` | Whether to cache frames for video export |
| `narrate_actions` | bool | `true` | Whether to generate action narration text |
| `max_recent_actions` | int | `5` | Number of recent history steps retained in observations |

---

### Step 3: Run Example

```bash
python launcher.py \
  --mode local \
  --manager-config manager/config.yaml \
  --env-config env/dwgym/dw_config.yaml \
  --llm-base-url <your-base-url> \
  --llm-api-key  <your-api-key> \
  --llm-model    <model-name> \
  --pool-size 1
```

---

### Step 4: Inspect results

After the run finishes, frame images are saved under `AIEvoBox/video/`. To enable this, set:

```yaml
env_params:
  use_vision: true
  capture_frames: true
```

> No frame images are generated when `use_vision: false`.

---

## Scenario List

DiscoveryWorld contains 8 scientific discovery topics, each with 3 difficulty levels:

| Scenario |
|----------|
| `"Proteomics"` |
| `"SpaceIllness"` |
| `"RosettaTranslation"` |
| `"Chemistry"` |
| `"RadioisotopeDating"` |
| `"ReactorLab"` |
| `"RocketScience"` |
| `"PlantNutrients"` |

> Scenario names should follow their actual definitions in the DiscoveryWorld source. If there is any mismatch, refer to [DiscoveryWorld GitHub](https://github.com/allenai/discoveryworld).

---

## Docker Deployment (environment tag TBD)

```bash
docker build -f env/dwgym/Dockerfile -t dwgym:latest .
```

- To use a prebuilt image, replace the registry with your own image repository, for example:

```bash
docker pull <your-registry>/dwgym:latest
```

---

## File Layout

```text
dwgym/
├── __init__.py
├── dw_env.py
├── dw_config.yaml
├── README.md
└── discoveryworld/      # generated after git clone, not committed to the repo
```

---

## Related Links

- [DiscoveryWorld GitHub](https://github.com/allenai/discoveryworld)
- [DiscoveryWorld paper](https://arxiv.org/abs/2402.03628)

## License

This adapter follows the AIEvoBox license. DiscoveryWorld has its own license; refer to its official documentation.
