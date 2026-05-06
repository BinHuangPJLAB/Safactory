# 配置

Safactory 配置分为三层：

1. 传给 `launcher.py` 的 CLI 参数。
2. `manager/config.yaml` 中的 manager 配置。
3. `env/` 下或通过 `--env-config` 传入的环境 YAML 文件。

CLI 参数优先级高于从 YAML 加载的值。

## 必要 CLI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | `local` | 执行模式：`local` 或 `remote`。 |
| `--env-transport` | `http` | 环境传输方式：`http` 或 `inproc`。 |
| `--env-config` | `None` | 单个环境 YAML 路径。与 `--env-root` 不兼容。 |
| `--env-root` | `env` | 未设置 `--env-config` 时递归扫描环境 YAML 的目录。 |
| `--pool-size` | `0` | 覆盖 YAML 中的 pool size。`0` 表示保留 YAML 值。 |
| `--llm-base-url` | 项目默认端点 | OpenAI 兼容模型端点。真实运行时请替换。 |
| `--llm-api-key` | `EMPTY` | API key。端点不需要 key 时可用 `EMPTY`。 |
| `--llm-model` | 项目默认模型 | 发送给端点的模型标识。 |
| `--max-steps` | `1000` | 每个 episode 的最大智能体步数。 |
| `--db-path` | `sqlite://android_envs.db` | SQLite 数据库 URI。 |
| `--log-dir` | `logs` | 运行日志目录。 |

示例：

```bash
python launcher.py \
  --mode local \
  --env-transport http \
  --env-config env/osgym/os_config.yaml \
  --llm-base-url http://YOUR_LLM_HOST/v1 \
  --llm-api-key YOUR_API_KEY \
  --llm-model YOUR_MODEL \
  --pool-size 1
```

## 完整 CLI 参考

| 类别 | 参数 | 默认值 | 说明 |
|------|------|--------|------|
| Job | `--job-id` | 为空时自动生成 | 记录在环境配置和 session 行中的标识符。 |
| Config | `--manager-config` | `./manager/config.yaml` | Manager YAML 路径。 |
| Config | `--exp-config` | `./core/exp/config.yaml` | 经验注入 YAML 路径。 |
| Storage | `--storage-type` | `sqlite` | 存储后端：`sqlite` 或 `cloud`。 |
| Storage | `--warmup-count` | `100` | 预加载到 manager 的环境配置数量。 |
| Storage | `--save-batch-size` | `100` | 存储环境配置的 batch 大小。 |
| Storage | `--disable-buffer` | buffer 启用 | 禁用带缓冲的记录写入。 |
| Storage | `--buffer-size` | `100` | 记录缓冲区容量。 |
| Storage | `--flush-interval` | `5.0` | 缓冲写入刷新间隔，单位秒。 |
| Storage | `--rebuild-table` / `--no-rebuild-table` | `false` | 加载配置前删除并重建 SQLite 表。 |
| Pool | `--multiplier` | `1.2` | 预热 `ceil(multiplier * pool_size)` 个 actor。 |
| Local service | `--start-local-upstream` / `--no-start-local-upstream` | `None` | 显式控制本地 FastAPI 服务启动。 |
| Local service | `--local-upstream-app` | `env.app:app` | 本地 HTTP 模式使用的 ASGI app。 |
| Local service | `--local-upstream-host` | `0.0.0.0` | 本地服务绑定 host。 |
| Local service | `--local-upstream-port` | `36663` | 本地服务端口。 |
| Local service | `--local-upstream-url` | `http://127.0.0.1:36663` | Runner 访问本地服务的 URL。 |
| Local service | `--wait-timeout` | `60.0` | 等待本地服务就绪的秒数。 |
| Interactor | `--message-cut` | `-1` | 上下文中保留的最近消息数量。`<= 0` 表示全部保留。 |
| Interactor | `--env-http-timeout-s` | `300.0` | 环境 HTTP 请求超时时间。 |
| Interactor | `--http-retries` | `2` | 环境调用失败后的重试次数。 |
| LLM | `--llm-temperature` | `0.3` | 采样温度。 |
| RL | `--rl-use-session-suffix-url` | `false` | 使用按 session 路由的 LLM proxy URL。 |
| RL | `--rl-group-size` | `0` | 覆盖每个 prompt 的重复样本数 `env_num`。 |
| RL | `--rl-epoch` | `1` | 为多个 RL epoch 复制环境配置。 |
| Logging | `--run-name` | 空 | 运行日志目录的可选前缀。 |
| Logging | `--console-log-level` | `INFO` | 控制台日志级别。 |
| Logging | `--file-log-level` | `DEBUG` | 文件日志级别。 |
| Logging | `--log-max-bytes` | `52428800` | 旧版轮转大小。在运行目录日志模式中忽略。 |
| Logging | `--log-backup-count` | `20` | 保留的最近运行日志目录数量。`0` 表示全部保留。 |
| Logging | `--debug-log` | `false` | 在控制台启用 debug 存储日志。 |

## Manager 配置

`manager/config.yaml` 控制数据库默认值、pool size、远程集群设置、环境启动命令和 RayJob 凭据。仓库中的值应视为示例，使用前请替换私有路径、镜像名和凭据。

最小本地模式示例：

```yaml
mode: local
pool_size: 1

database:
  driver: sqlite
  sqlite_path: android_envs.db

cluster:
  http:
    port: 36663
    timeout_s: 100
    concurrency: 200
    startup_concurrency: 16
```

远程模式会增加按环境划分的集群启动设置：

```yaml
cluster:
  env_types:
    os_gym:
      quotagroup: "your-quota"
      entrypoint: "bash /path/to/start_os_gym.sh"
      resources:
        head:
          cpu: 18
          gpu: 0
          memory: 72Gi
          privileged: true
      limit: 12

rayjob:
  domain: "https://your-rayjob-platform.example"
  tenant: "your-tenant"
  access_key: "replace-me"
  secret_key: "replace-me"
  project: "your-project"
```

## 环境 YAML

每个环境 YAML 定义一个或多个可运行环境组。

```yaml
environments:
  - env_name: os_gym
    env_image: your_optional_runtime_image
    env_num: 1
    dataset: path/to/cases.jsonl
    dataset_load_mode: eager
    env_params:
      max_steps: 30
      prompt_format: kimi
```

| 字段 | 必需 | 说明 |
|------|------|------|
| `env_name` | 是 | 由 `env/env_factory.py` 解析的注册名。 |
| `env_image` | 否 | 远程 launcher 使用的运行时镜像。 |
| `env_num` | 否 | 该环境组的并行实例数。 |
| `dataset` | 否 | 数据加载器消费的 JSON、JSONL、YAML、文本或 parquet 路径。 |
| `dataset_load_mode` | 否 | 默认 `eager`；部分基于 parquet 的环境使用 `parquet_row_ref`。 |
| `env_params` | 否 | 传入环境构造函数的关键字参数。 |

支持的注册名包括 `android_gym`、`os_gym`、`mc`、`mc_gym`、`embodied_alfred`、`emb`、`qa_gym`、`dabstepgym`、`discoveryworld`、`deepeyes_env`、`geo3k_vl_test`、`robotrustbench`、`math500_text` 和 `math500`。

## 数据路径

相对数据集路径会从 YAML 文件所在目录解析。例如 `env/androidgym/android_env.yaml` 中的 `dataset: cases.jsonl` 会解析为 `env/androidgym/cases.jsonl`。

共享存储、Docker 挂载卷或外部下载的数据集请使用绝对路径。
