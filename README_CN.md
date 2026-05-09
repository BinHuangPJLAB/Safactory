<div align="center">

# Safactory

<p align="center">
    中文 &nbsp ｜ &nbsp <a href="README.md">English</a>
</p>

**测训一体的下一代智能体基础设施，支持在 OS、Android、Minecraft、具身智能、QA、数据处理、科学发现等多类环境中评测智能体、采集轨迹，并进行强化学习训练。首次验证智能体可信Scaling Law，实现安全能力提升且无对齐税。**

[快速开始](#quick-start) |
[演示](#demo) |
[环境](docs/environments_CN.md) |
[RL 训练](docs/rl-training_CN.md) |
[自定义环境](docs/custom-environment_CN.md) |
[配置](docs/configuration_CN.md) |
[数据](docs/data-manager_CN.md) |
[报告](report.pdf)

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Execution](https://img.shields.io/badge/mode-local%20%7C%20remote-orange)
![LLM](https://img.shields.io/badge/LLM-OpenAI--compatible-purple)

</div>

---

## <a id="why-safactory"></a>✨ 为什么使用 Safactory

![tax](fig/tax.png)

Safactory 是面向需要统一完成评测、数据生成和 RL 训练的团队的智能体沙箱。它提供统一的环境接口、并发 rollout 管理、OpenAI 兼容模型访问、轨迹持久化，以及面向 Slime / GRPO 训练的 Buffer Server 桥接。

| 需求 | Safactory 提供 |
|------|----------------|
| 评测智能体 | 在真实交互环境中运行 LLM 或 VLM 智能体并收集奖励。 |
| 构建轨迹数据 | 将消息、动作、观察、奖励和环境状态持久化到 SQLite。 |
| RL 训练 | 通过内置 Buffer Server 将 rollout 轨迹流式送入 Slime。 |
| 添加新环境 | 通过标准接口接入新的环境。 |

核心能力：

- 多领域环境：OS、Android、Minecraft、RoboTrustBench、Embodied ALFRED、QA、DABStep、DiscoveryWorld、DeepEyes、Geo3K-VL 和 Math500。
- 通过环境池和异步 worker 支持高并发 rollout。
- 支持 vLLM、SGLang、托管 API 和本地代理等 OpenAI 兼容模型服务。
- 支持本地单机模式和基于 RayJob 的远程集群模式。
- 可选的经验抽取和 prompt 时经验注入。

## <a id="demo"></a>🎬 演示

<div align="center">

https://github.com/user-attachments/assets/4c551b27-ce4d-4fc8-8df6-d6dc8100cc88

*点击播放查看完整演示*

</div>

## <a id="quick-start"></a>🚀 快速开始

### 安装

```bash
git clone https://github.com/AI45Lab/Safactory.git
cd Safactory
pip install -r requirements.txt
```

部分环境有额外运行时依赖。运行 Docker、模拟器、虚拟机或仿真器任务前，请先查看[支持的环境](docs/environments_CN.md)。

### 评测模型

```bash
python launcher.py \
  --env-config env/osgym/os_config.yaml \   # 选择评测环境（OS / Android / Minecraft 等）
  --llm-base-url http://YOUR_LLM_HOST/v1 \  # 模型服务地址
  --llm-api-key YOUR_API_KEY \              # API Key
  --llm-model YOUR_MODEL \                  # 模型名称
  --pool-size 500                           # 并发智能体实例数
```

该命令会启动 runner，加载选定的环境配置，调度任务，调用模型端点，并将 step 级记录写入 SQLite。

### 采集轨迹数据

每次 rollout 都会自动记录。默认 CLI 数据库路径为 `sqlite://env_trajs.db`；可以用 `--db-path` 覆盖：

```bash
python launcher.py \
  --env-config env/osgym/os_config.yaml \
  --db-path sqlite://runs/os_eval.db \
  --llm-base-url http://YOUR_LLM_HOST/v1 \
  --llm-api-key YOUR_API_KEY \
  --llm-model YOUR_MODEL
```

表结构和查询示例见[数据管理器](docs/data-manager_CN.md)。

### 使用 RL 训练

Safactory 通过 Buffer Server 与 [Slime](https://github.com/THUDM/slime) 集成：

```bash
# 终端 1：Slime 训练进程
cd rl
./run_slime_generator_vl.sh

# 终端 2：Safactory Buffer Server 和 rollout runner
cd rl
./run_buffer_server.sh
```

完整说明见 [RL 训练](docs/rl-training_CN.md)。

## <a id="datasets"></a>📦 数据集

Safactory 可以生成可复用的轨迹数据集。公开 OS 轨迹发布在 Hugging Face：

- [AI45Research/SATraj-OS](https://huggingface.co/datasets/AI45Research/SATraj-OS)，一个由 Safactory 生成、用于智能体训练和分析的 OS 轨迹数据集。

Safactory 生成的数据也支持智能体安全训练。本实验使用 SATraj-OS 对 8B 视觉语言智能体进行微调，得到 **SATraj-Agent-8B**，并在 OS-Harm 上评估安全性、在 OSWorld 上评估任务能力。结果显示，SATraj-Agent-8B 将平均 Unsafe 从 31.33% 降至 **3.33%**，同时将 OSWorld Total 从 14.40% 提升至 22.16%，说明安全能力提升没有带来安全对齐税。

<table>
  <thead>
    <tr>
      <th rowspan="2">模型</th>
      <th colspan="7">安全 (OS-Harm)</th>
      <th colspan="5">能力 (OSWorld，越高越好)</th>
    </tr>
    <tr>
      <th>平均 Unsafe ↓</th>
      <th>误用 Unsafe ↓</th>
      <th>误用 Completed ↓</th>
      <th>注入 Unsafe ↓</th>
      <th>注入 Completed ↑</th>
      <th>失控 Unsafe ↓</th>
      <th>失控 Completed ↑</th>
      <th>Total</th>
      <th>Chrome</th>
      <th>GIMP</th>
      <th>OS</th>
      <th>VS Code</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Qwen3.5-397B</td><td align="right">32.00%</td><td align="right">62.00%</td><td align="right">8.00%</td><td align="right">16.00%</td><td align="right">40.00%</td><td align="right">18.00%</td><td align="right">6.00%</td><td align="right"><strong>62.20%</strong></td><td align="right">-</td><td align="right">-</td><td align="right">-</td><td align="right">-</td></tr>
    <tr><td>Qwen3vl-8b</td><td align="right">31.33%</td><td align="right">69.33%</td><td align="right">22.67%</td><td align="right">10.00%</td><td align="right">14.00%</td><td align="right">14.67%</td><td align="right">4.00%</td><td align="right">14.40%</td><td align="right">28.26%</td><td align="right">15.38%</td><td align="right">25.00%</td><td align="right">21.74%</td></tr>
    <tr><td>SATraj-Agent-8B</td><td align="right"><strong>3.33%</strong></td><td align="right"><strong>0.00%</strong></td><td align="right"><strong>0.00%</strong></td><td align="right"><strong>8.00%</strong></td><td align="right"><strong>54.00%</strong></td><td align="right"><strong>2.00%</strong></td><td align="right"><strong>10.00%</strong></td><td align="right">22.16%</td><td align="right"><strong>34.78%</strong></td><td align="right"><strong>42.31%</strong></td><td align="right"><strong>29.17%</strong></td><td align="right"><strong>56.52%</strong></td></tr>
  </tbody>
</table>

## <a id="documentation"></a>📚 文档

| 指南 | 内容 |
|------|------|
| [配置](docs/configuration_CN.md) | CLI 参数、manager YAML 和环境 YAML 格式。 |
| [支持的环境](docs/environments_CN.md) | 环境注册名、前置依赖和安装链接。 |
| [数据管理器](docs/data-manager_CN.md) | SQLite 表结构、存储行为和查询示例。 |
| [RL 训练](docs/rl-training_CN.md) | Slime 集成、Buffer Server 设置和 RL 变量。 |
| [自定义环境](docs/custom-environment_CN.md) | 最小 `BaseEnv` 实现和注册流程。 |
| [经验抽取与注入](docs/experience-extraction-injection_CN.md) | 将历史轨迹作为 prompt 时经验复用。 |

## <a id="architecture"></a>🏗️ 架构

![Safactory architecture](fig/overview.png)

整体上，`launcher.py` 会加载环境 YAML 文件，启动或连接环境服务，将观察发送到 OpenAI 兼容模型端点，通过数据管理器记录每次交互，并可选择将完成的 rollout 转发给 RL 训练。

## <a id="contributing"></a>🤝 贡献

欢迎贡献新环境、bug 修复、文档改进和可复现实例。

1. Fork 仓库。
2. 在 `env/<name>/` 下添加或更新环境。
3. 包含 YAML 配置和简短 README，说明环境特定依赖。
4. 使用 `launcher.py` 运行本地 smoke test。
5. 打开 pull request，并说明安装步骤和预期行为。

## <a id="citation"></a>📝 引用

如果 Safactory 或 Safactory 生成的数据集对你的工作有帮助，请引用本仓库以及你使用的具体数据集或报告。

```bibtex
@misc{safactory,
  title = {Safactory: A Universal AI Agent Sandbox for Evaluation, Data Construction, and RL Training},
  howpublished = {\url{https://github.com/AI45Lab/Safactory}},
  year = {2026}
}
```
