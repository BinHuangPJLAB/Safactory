<div align="center">

# Safactory

<p align="center">
    <a href="README_CN.md">中文</a> &nbsp ｜ &nbsp English
</p>

**A next-generation agent infrastructure that integrates evaluation and training, supporting agent evaluation, trajectory collection, and reinforcement learning training across multiple types of environments including OS, Android, Minecraft, embodied AI, QA, data processing, and scientific discovery. It is the first to validate a trustworthy scaling law for agents, achieving improved safety capabilities without an alignment tax.**

[Quick Start](#quick-start) |
[Demo](#demo) |
[Environments](docs/environments.md) |
[RL Training](docs/rl-training.md) |
[Custom Environments](docs/custom-environment.md) |
[Configuration](docs/configuration.md) |
[Data](docs/data-manager.md) |
[Report](report.pdf)

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Execution](https://img.shields.io/badge/mode-local%20%7C%20remote-orange)
![LLM](https://img.shields.io/badge/LLM-OpenAI--compatible-purple)

</div>

---

## <a id="why-safactory"></a>✨ Why Safactory

Safactory is an agent sandbox for teams that need one pipeline for evaluation, data generation, and RL training. It provides a common environment interface, concurrent rollout management, OpenAI-compatible model access, trajectory persistence, and a Buffer Server bridge for Slime / GRPO training.

| Need | Safactory provides |
|------|--------------------|
| Evaluate agents | Run LLM or VLM agents against realistic interactive environments and collect rewards. |
| Build trajectory data | Persist messages, actions, observations, rewards, and environment state to SQLite. |
| Train with RL | Stream rollout trajectories into Slime through the built-in Buffer Server. |
| Add new Env | Access new environments through standard interfaces. |

Core features:

- Multi-domain environments: OS, Android, Minecraft, RoboTrustBench, Embodied ALFRED, QA, DABStep, DiscoveryWorld, DeepEyes, Geo3K-VL, and Math500.
- High-concurrency rollouts through environment pools and async workers.
- OpenAI-compatible model integration for vLLM, SGLang, hosted APIs, and local proxies.
- Local single-machine mode and remote RayJob-backed cluster mode.
- Optional experience extraction and prompt-time experience injection.

## <a id="demo"></a>🎬 Demo

<div align="center">

https://github.com/user-attachments/assets/4c551b27-ce4d-4fc8-8df6-d6dc8100cc88

*点击播放查看完整演示*

</div>

## <a id="quick-start"></a>🚀 Quick Start

### Install

```bash
git clone https://github.com/AI45Lab/Safactory.git
cd Safactory
pip install -r requirements.txt
```

Some environments have extra runtime dependencies. See [Supported Environments](docs/environments.md) before running Docker, emulator, VM, or simulator-backed tasks.

### Evaluate a model

```bash
python launcher.py \
  --env-config env/osgym/os_config.yaml \   # Select the evaluation environment (OS / Android / Minecraft, etc.)
  --llm-base-url http://YOUR_LLM_HOST/v1 \  # Model service address
  --llm-api-key YOUR_API_KEY \              # API Key
  --llm-model YOUR_MODEL \                  # Model name
  --pool-size 500                           # Number of concurrent agent instances
```

This starts the runner, loads the selected environment configuration, schedules tasks, calls the model endpoint, and writes step-level records to SQLite.

### Collect trajectory data

Every rollout is recorded automatically. The default CLI database path is `sqlite://env_trajs.db`; override it with `--db-path`:

```bash
python launcher.py \
  --env-config env/osgym/os_config.yaml \
  --db-path sqlite://runs/os_eval.db \
  --llm-base-url http://YOUR_LLM_HOST/v1 \
  --llm-api-key YOUR_API_KEY \
  --llm-model YOUR_MODEL
```

See [Data Manager](docs/data-manager.md) for schema details and query examples.

### Train with RL

Safactory integrates with [Slime](https://github.com/THUDM/slime) through a Buffer Server:

```bash
# Terminal 1: Slime training process
cd rl
./run_slime_generator_vl.sh

# Terminal 2: Safactory Buffer Server and rollout runner
cd rl
./run_buffer_server.sh
```

Full instructions are in [RL Training](docs/rl-training.md).

## <a id="datasets"></a>📦 Datasets

Safactory can generate reusable trajectory datasets. The public OS trajectory release is available on Hugging Face:

- [AI45Research/SATraj-OS](https://huggingface.co/datasets/AI45Research/SATraj-OS), a Safactory-generated OS trajectory dataset for agent training and analysis.

Safactory-generated data also supports safe agent training. Our results show a safe scaling law for agents: safety improves with training scale while task capability is preserved, so the framework can improve agent safety without a safety alignment tax.

<table>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th colspan="4">Ability (OSWorld)</th>
      <th colspan="6">Safety (RiOSWorld)</th>
    </tr>
    <tr>
      <th>Total</th>
      <th>Chrome</th>
      <th>OS</th>
      <th>VS Code</th>
      <th>Total</th>
      <th>Text</th>
      <th>Web</th>
      <th>Office</th>
      <th>Note</th>
      <th>Reddit</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>claude-opus-4.6-thinking</td><td align="right">31.25</td><td align="right">35.00</td><td align="right"><strong>55.00</strong></td><td align="right">15.00</td><td align="right">48.58</td><td align="right">72.41</td><td align="right">35.71</td><td align="right"><strong>63.64</strong></td><td align="right"><strong>100.00</strong></td><td align="right">76.92</td></tr>
    <tr><td>qwen3.5-plus</td><td align="right">20.00</td><td align="right">35.00</td><td align="right">25.00</td><td align="right">5.00</td><td align="right">51.94</td><td align="right">75.86</td><td align="right"><strong>42.86</strong></td><td align="right"><strong>63.64</strong></td><td align="right">86.96</td><td align="right">80.00</td></tr>
    <tr><td>kimi-k2.5</td><td align="right"><strong>33.33</strong></td><td align="right"><strong>52.63</strong></td><td align="right">50.00</td><td align="right"><strong>20.00</strong></td><td align="right">32.17</td><td align="right">27.59</td><td align="right">23.81</td><td align="right">59.09</td><td align="right">30.43</td><td align="right"><strong>100.00</strong></td></tr>
    <tr><td>glm-4.6v</td><td align="right">10.13</td><td align="right">10.53</td><td align="right">15.00</td><td align="right">5.00</td><td align="right"><strong>54.09</strong></td><td align="right"><strong>85.71</strong></td><td align="right">35.71</td><td align="right">2.73</td><td align="right">91.30</td><td align="right">93.33</td></tr>
    <tr><td>step-3</td><td align="right">18.82</td><td align="right">5.26</td><td align="right">45.00</td><td align="right">15.00</td><td align="right">32.68</td><td align="right">39.29</td><td align="right">33.33</td><td align="right">0.00</td><td align="right">42.86</td><td align="right">90.00</td></tr>
    <tr><td>intern-s1-pro</td><td align="right">14.28</td><td align="right">15.00</td><td align="right">31.58</td><td align="right">0.00</td><td align="right">38.82</td><td align="right">58.62</td><td align="right">33.33</td><td align="right">4.76</td><td align="right">66.67</td><td align="right">86.67</td></tr>
  </tbody>
</table>

## <a id="documentation"></a>📚 Documentation

| Guide | What it covers |
|-------|----------------|
| [Configuration](docs/configuration.md) | CLI flags, manager YAML, and environment YAML format. |
| [Supported Environments](docs/environments.md) | Environment registry names, prerequisites, and setup links. |
| [Data Manager](docs/data-manager.md) | SQLite schema, storage behavior, and query examples. |
| [RL Training](docs/rl-training.md) | Slime integration, Buffer Server setup, and RL variables. |
| [Custom Environment](docs/custom-environment.md) | Minimal `BaseEnv` implementation and registration flow. |
| [Experience Extraction and Injection](docs/experience-extraction-injection.md) | Reusing historical trajectories as prompt-time experience. |

## <a id="architecture"></a>🏗️ Architecture

![Safactory architecture](fig/overview.png)

At a high level, `launcher.py` loads environment YAML files, starts or connects to environment services, sends observations to an OpenAI-compatible model endpoint, records every interaction through the data manager, and optionally forwards completed rollouts to RL training.

## <a id="contributing"></a>🤝 Contributing

Contributions are welcome for new environments, bug fixes, documentation improvements, and reproducible examples.

1. Fork the repository.
2. Add or update an environment under `env/<name>/`.
3. Include a YAML config and a short README for environment-specific dependencies.
4. Run a local smoke test with `launcher.py`.
5. Open a pull request with the setup notes and expected behavior.

## <a id="citation"></a>📝 Citation

If Safactory or Safactory-generated datasets are useful in your work, cite the repository and the specific dataset or report you used.

```bibtex
@misc{safactory,
  title = {Safactory: A Universal AI Agent Sandbox for Evaluation, Data Construction, and RL Training},
  howpublished = {\url{https://github.com/AI45Lab/Safactory}},
  year = {2026}
}
```
