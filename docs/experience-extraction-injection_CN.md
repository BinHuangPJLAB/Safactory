# 经验抽取与注入

Safactory 可以将历史轨迹作为 prompt 时经验复用。工作流分为两个阶段：

1. 从已记录轨迹中抽取可复用经验。
2. 将相关经验注入到后续 episode 中。

## 文件

| 文件或目录 | 用途 |
|------------|------|
| `exp_service/exp_service_config.example.yaml` | 抽取服务配置示例。 |
| `core/exp/config.yaml` | `launcher.py` 使用的运行时 prompt 注入配置。 |
| `experiences/` | 建议的经验输出目录。 |

## 抽取经验

在 Safactory 数据库中已有轨迹数据后运行抽取：

```bash
python3 -m exp_service run-once --config exp_service/exp_service_config.example.yaml
```

真实实验前请先复制示例配置，然后设置源数据库、输出目录、模型端点和抽取选项。

## 启用 Prompt 注入

更新 `core/exp/config.yaml`：

```yaml
enabled: true
dir: ./experiences
top_k: 3
mode: template
embedding_model: null
```

支持的模式：

| 模式 | 行为 |
|------|------|
| `template` | 选择高排名的经验模板。 |
| `ucb` | 使用带反馈的简单 bandit 策略。 |
| `contextual_ucb` | 使用任务文本和反馈历史选择经验。 |

设置 `enabled: false` 可关闭注入。

## 带注入运行

`launcher.py` 默认读取 `core/exp/config.yaml`。可以用 `--exp-config` 指向其他文件：

```bash
python3 launcher.py \
  --env-config env/osgym/os_config.yaml \
  --exp-config ./core/exp/config.yaml \
  --llm-base-url http://YOUR_LLM_HOST/v1 \
  --llm-api-key YOUR_API_KEY \
  --llm-model YOUR_MODEL
```

## 实用建议

- 保持抽取输出目录和注入 `dir` 一致。
- 先使用 `mode: template`，行为更可预测。
- 不同实验使用不同经验目录，避免反馈统计混合。
- 将经验库与生成它的数据集或模型运行一起做版本管理。
