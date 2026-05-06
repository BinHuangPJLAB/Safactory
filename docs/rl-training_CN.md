# 强化学习训练

Safactory 通过 Buffer Server 与 [Slime](https://github.com/THUDM/slime) 集成。Safactory 采集 rollout 轨迹，Buffer Server 将完成的轨迹组成 batch，Slime 再拉取 batch 进行 GRPO 风格训练。

## 架构

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

只要端口可达，rollout 进程和训练进程可以运行在不同机器上。

## 前置条件

- 已通过 `pip install -r requirements.txt` 安装 Safactory 依赖。
- 训练进程已安装 Slime。
- 至少有一个可成功运行的 Safactory 环境配置。
- 分布式运行时，Buffer Server、LLM proxy 和 Slime 进程之间网络可达。

可以使用 Slime 仓库中的上游构建脚本或官方 Docker 流程安装 Slime。

## 配置

在 shell 中导出 RL 变量，或放在本地 `.env` 文件中由启动脚本 source。当前 checkout 没有提交 `rl/.env.example`。

当前 Buffer Server 读取旧的 `AIEVOBOX_*` 变量前缀；除非代码已更新，否则请使用这些变量名。

## 启动训练

终端 1，Slime 进程：

```bash
cd rl

# 纯文本模型
./run_slime_generator.sh

# 视觉语言模型
./run_slime_generator_vl.sh
```

终端 2，Buffer Server 和 Safactory rollout runner：

```bash
cd rl
./run_buffer_server.sh
```

## 关键变量

环境选择：

| 变量 | 说明 |
|------|------|
| `AIEVOBOX_ROOT` | Buffer Server 用于定位 `launcher.py` 的 Safactory 仓库根目录。 |
| `AIEVOBOX_ENV_CONFIG` | 面向单环境运行的环境 YAML 路径。 |
| `AIEVOBOX_ENV_ROOT` | 扫描多个环境 YAML 文件的目录。 |
| `AIEVOBOX_ENV_TRANSPORT` | Launcher 环境传输方式，默认 `http`。 |
| `AIEVOBOX_DB_URL` | RL rollout 存储使用的 SQLite URI。 |
| `AIEVOBOX_POOL_SIZE` | Runner pool 大小。 |
| `AIEVOBOX_MAX_STEPS` | 最大 episode 步数。 |
| `AIEVOBOX_MESSAGE_CUT` | 消息历史保留长度。 |

训练：

| 变量 | 说明 |
|------|------|
| `RL_GROUP_SIZE` | 每个 prompt 的采样数，映射到 Slime 采样设置。 |
| `RL_EPOCH` | rollout epoch 数。 |
| `RL_OFF_BY_N` | 模型权重版本允许的最大滞后。`0` 表示严格 on-policy。 |
| `SLIME_GLOBAL_BATCH_SIZE` | 全局训练 batch size。 |

服务：

| 服务 | 默认端口 | 覆盖变量 |
|------|----------|----------|
| Buffer Server | `18889` | `BUFFER_SERVER_PORT` |
| LLM Proxy | `18890` | `LLM_PROXY_PORT` |

## 说明

- Buffer Server 所在机器需要 Safactory 依赖，但不需要 Slime。
- Slime 训练机器需要 Slime，并且需要能访问 Buffer Server。
- 扩容前先用一个环境和较小的 `RL_GROUP_SIZE` 启动。
- 在 `.env` 中明确写出模型端点、数据集路径和输出目录，便于复现。
