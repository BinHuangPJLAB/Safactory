# Custom Environment

Safactory environments implement a small `BaseEnv` contract and are loaded by registry name. A minimal environment can be added with one Python class, one lazy importer, and one YAML config.

## 1. Implement `BaseEnv`

Create a module such as `env/mygym/my_env.py`:

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

## Interface Contract

| Method | Required | Purpose |
|--------|----------|---------|
| `get_task_prompt()` | yes | Initial chat messages for the agent. |
| `reset(seed=None)` | yes | Reset the environment and return the first observation. |
| `step(action)` | yes | Execute one action and return `(obs, reward, done, truncated, info)`. |
| `render()` | no | Return a frame or screenshot for visual logs. |
| `close()` | no | Release external resources. |

## 2. Register the Environment

Safactory uses lazy imports so heavyweight environment dependencies are loaded only when needed.

Add an importer in `env/registry.py`:

```python
def _import_my_env() -> Type:
    from env.mygym.my_env import MyEnv
    return MyEnv
```

Expose the registry key in `env/env_factory.py`:

```python
from env.registry import _import_my_env

_ENV_IMPORTERS = {
    "my_env": _import_my_env,
}
```

The key must match both the `@register_env("my_env")` decorator and the YAML `env_name`.

## 3. Write a YAML Config

Create `env/mygym/my_config.yaml`:

```yaml
environments:
  - env_name: my_env
    env_num: 1
    dataset: cases.jsonl
    env_params:
      my_param: value
```

Dataset rows are merged into `env_params` under the `dataset` key. Supported dataset formats include JSON arrays, JSONL, YAML arrays, plain text, and parquet for environments that use parquet row references.

## 4. Run a Test

```bash
python launcher.py \
  --mode local \
  --env-config env/mygym/my_config.yaml \
  --llm-base-url http://YOUR_LLM_HOST/v1 \
  --llm-api-key YOUR_API_KEY \
  --llm-model YOUR_MODEL \
  --pool-size 1
```

## Tips

- Keep `env_num: 1` while developing a new environment.
- Add a small dataset row first, then scale to full benchmark data.
- Implement `render()` when visual debugging is useful.
- Keep optional external credentials and large data paths in YAML, not in the environment class.
- Use existing adapters such as `env/geo3k_vl/`, `env/deepeyes/`, and `env/osgym/` as references.
