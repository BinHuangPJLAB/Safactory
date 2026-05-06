# 自定义环境

Safactory 环境实现一个很小的 `BaseEnv` 契约，并按注册名加载。一个最小环境可以由一个 Python 类、一个懒加载 importer 和一个 YAML 配置组成。

## 1. 实现 `BaseEnv`

创建类似 `env/mygym/my_env.py` 的模块：

```python
from core.env.base_env import BaseEnv
from core.env.env_register import register_env


@register_env("my_env")
class MyEnv(BaseEnv):
    def __init__(self, my_param: str = "default", **kwargs):
        super().__init__(**kwargs)
        self.my_param = my_param

    def observation_space(self) -> str:
        return "Describe the observable state."

    def action_space(self) -> str:
        return "Describe the valid action format."

    def get_task_prompt(self) -> list:
        return [{"role": "user", "content": "Complete the task."}]

    def reset(self, seed=None):
        return "initial observation"

    def step(self, action: str):
        obs = "next observation"
        reward = 0.0
        done = False
        truncated = False
        info = {}
        return obs, reward, done, truncated, info

    def render(self):
        return None

    def close(self):
        pass
```

## 接口契约

| 方法 | 必需 | 用途 |
|------|------|------|
| `get_task_prompt()` | 是 | 给智能体的初始聊天消息。 |
| `reset(seed=None)` | 是 | 重置环境并返回第一个观察。 |
| `step(action)` | 是 | 执行一个动作并返回 `(obs, reward, done, truncated, info)`。 |
| `render()` | 否 | 返回用于视觉日志的帧或截图。 |
| `close()` | 否 | 释放外部资源。 |

## 2. 注册环境

Safactory 使用懒加载导入，因此重型环境依赖只会在需要时加载。

在 `env/registry.py` 中添加 importer：

```python
def _import_my_env() -> Type:
    from env.mygym.my_env import MyEnv
    return MyEnv
```

在 `env/env_factory.py` 中暴露注册 key：

```python
from env.registry import _import_my_env

_ENV_IMPORTERS = {
    "my_env": _import_my_env,
}
```

该 key 必须同时匹配 `@register_env("my_env")` 装饰器和 YAML 中的 `env_name`。

## 3. 编写 YAML 配置

创建 `env/mygym/my_config.yaml`：

```yaml
environments:
  - env_name: my_env
    env_num: 1
    dataset: cases.jsonl
    env_params:
      my_param: value
```

数据集行会合并到 `env_params` 下的 `dataset` key。支持的数据集格式包括 JSON 数组、JSONL、YAML 数组、纯文本，以及供使用 parquet 行引用的环境使用的 parquet。

## 4. 运行测试

```bash
python launcher.py \
  --mode local \
  --env-config env/mygym/my_config.yaml \
  --llm-base-url http://YOUR_LLM_HOST/v1 \
  --llm-api-key YOUR_API_KEY \
  --llm-model YOUR_MODEL \
  --pool-size 1
```

## 建议

- 开发新环境时先保持 `env_num: 1`。
- 先添加一条小数据集样本，再扩展到完整 benchmark 数据。
- 视觉调试有用时实现 `render()`。
- 将可选外部凭据和大数据路径放在 YAML 中，不要写进环境类。
- 参考 `env/geo3k_vl/`、`env/deepeyes/` 和 `env/osgym/` 等现有 adapter。
