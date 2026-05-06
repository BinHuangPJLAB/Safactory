# EmbodiedGym - Alfred Environment Adapter

Integrates the EmbodiedBench Alfred environment into AIEvoBox.

## 📁 File Layout

```text
embodiedgym/
├── __init__.py                 # Module initialization
├── embodied_env.py            # Core adapter class
├── embodied_config.yaml       # Environment config file
├── test_embodied_env.py       # Test script
└── README.md                  # This document
```

## 🚀 Quick Start

### 1. Prerequisites

Install and configure the environment with the following steps:

```bash
# 1. Clone the AIEvoBox repository
git clone https://gitee.pjlab.org.cn/L2/safeai/kilab/AIEvoBox.git

# 2. Enter the embodiedgym directory
# Note: the cloned directory name may be AIEvoBox or AIEvoBox-new
cd AIEvoBox/env/embodiedgym

# 3. Clone EmbodiedBench
git clone git@github.com:EmbodiedBench/EmbodiedBench.git

# 4. Create and activate the Conda environment
# Enter the EmbodiedBench source directory; adjust the name if it was cloned as EmbodiedBench-master
cd EmbodiedBench-master
conda env create -f conda_envs/envir.yaml --prefix /root/envs/embench1
conda activate /root/envs/embench1
pip install -e .

# 5. Install system dependencies
sudo apt-get update
sudo apt-get install -y xvfb
# Start Xvfb
Xvfb :1 -screen 0 1024x768x24 &

# Install fonts
apt-get update
apt-get install -y fonts-ubuntu

# 6. Configure AI2THOR resources
# Copy predownloaded AI2THOR data
cp -r /mnt/shared-storage-user/evobox-share/gaozhenkun/gzk/ai2thor/.ai2thor ~/.ai2thor
```

### 2. Prepare Data

Download and configure the EB-ALFRED dataset:

```bash
# Make sure the environment is active
conda activate embench1

# Download the dataset
git clone https://huggingface.co/datasets/EmbodiedBench/EB-ALFRED

# Move data to the expected directory
# Note: this assumes the current directory is EmbodiedBench-master
mv EB-ALFRED embodiedbench/envs/eb_alfred/data/json_2.1.0
```

### 3. Start the VLLM Service

Make sure the VLLM service is running with the Qwen2.5-VL-7B-Instruct model:

```bash
vllm serve /mnt/shared-storage-user/steai-share/hf-hub/Qwen2.5-VL-7B-Instruct \
    --dtype half \
    --port 8001 \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --max_model_len 30000
```

> **Note**: After starting the service, update `baseurl` and `port` in `examples/run_8_trading_envs.sh` to the actual service address.

### 4. Run Tests

Check whether the environment works:

```bash
cd /AIEvoBox/env/embodiedgym
python test_embodied_env.py
```

### 5. Run Examples

#### Option 1: Python script

```bash
cd /AIEvoBox
python examples/multi_env_example.py
```

#### Option 2: Shell script

```bash
cd AIEvoBox
bash examples/run_8_embodied_envs1.sh
```

## ⚙️ Configuration

### Environment Parameters

The following parameters can be configured in `embodied_config.yaml`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `eval_set` | str | 'base' | Evaluation set name<br>Options: 'base', 'common_sense', 'complex_instruction', 'spatial', 'visual_appearance', 'long_horizon' |
| `down_sample_ratio` | float | 1.0 | Data sampling ratio<br>Range: 0.0-1.0; 1.0 means all data |
| `resolution` | int | 500 | Image resolution in pixels<br>Recommended range: 300-800 |
| `detection_box` | bool | false | Whether to display object detection boxes on images |
| `max_episode_steps` | int | 30 | Maximum steps per episode |
| `max_invalid_actions` | int | 10 | Maximum consecutive invalid actions |
| `exp_name` | str | 'aievobox_alfred' | Experiment name for logs and result storage |

### Example Config

```yaml
environments:
  - env_name: embodied_alfred
    env_num: 2
    env_params:
      eval_set: "base"
      down_sample_ratio: 0.1  # Use 10% of data for a quick test
      resolution: 500
      max_episode_steps: 30
      exp_name: "test_base"
```

## 🔗 Related Links

- [EmbodiedBench GitHub](https://github.com/embodied-generalist/embodiedbench)
- [ALFRED Dataset](https://askforalfred.com/)
- [AI2THOR](https://ai2thor.allenai.org/)
- [VLLM](https://github.com/vllm-project/vllm)

## 📄 License

This adapter follows the AIEvoBox license. EmbodiedBench and ALFRED have their own licenses; refer to their official documentation.
