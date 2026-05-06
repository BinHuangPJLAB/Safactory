## Goal

Build a Docker image on top of the official Ray base image. After startup, the image should run a FastAPI service that hosts the `MCGym` environment. Overall requirements:

- Fully install the `env/mc` environment, including `pip install -e .` for `MineStudio`.
- Remove hard-coded local absolute paths in `malmo.py`, `entry.py`, and `utils/temp.py`.
- Properly place or mount `offline-mcprec-6.13.jar` inside the image.
- FastAPI provides interfaces such as `reset`, `step`, `render`, and `close`.

The following sections describe the design and notes step by step so they can be discussed and adjusted incrementally.

## 1. Run and Test First

- Build, with optional offline wheels:

  ```bash
  docker build --no-cache -f env/mc/Dockerfile.mc-env -t mc-fastapi \
    --build-arg USE_LOCAL_WHEELS=true \
    --build-arg http_proxy="$http_proxy" \
    --build-arg https_proxy="$https_proxy" \
    --build-arg no_proxy="$no_proxy" \
    .
  ```

- Run, CPU/headless by default:

  `docker run --rm -p 8000:8000 mc-fastapi`.
  If port 8000 is occupied, use `-p 8010:8000` or stop `nginx` first.

- Service tests. Replace `localhost` with the real IP when others need access. All APIs use JSON:

  1. Health check

     ```bash
     curl http://localhost:8000/healthz
     ```

     Returns fields such as `env_ready`, whether the environment has been reset, `active_config`, and `jar_path`.

  2. Initialize environment. Parameters are optional.

     ```bash
     curl -X POST http://localhost:8000/reset \
       -H 'Content-Type: application/json' \
       -d '{"env_config":"env/mc/config/collect/collect_wood.yaml","seed":123,"force_recreate":true}'
     ```

     - `env_config`: task YAML relative to `PROJECT_ROOT`.
     - `seed`: Minecraft world seed.
     - `force_recreate`: when `true`, force-close and recreate the environment instance. Default is `false`.

  3. Send an action. Strings and dictionaries are both supported; the example uses a string for LLM output.

     ```bash
     curl -X POST http://localhost:8000/step \
       -H 'Content-Type: application/json' \
       -d '{"action":"[{\"action\":\"forward\",\"camera\":{\"yaw\":5.0,\"pitch\":-3.0},\"attack\":1,\"jump\":0}]"}'
     ```

     You can also pass a dictionary: `{"action":{"buttons":{"attack":1,"forward":1}, "camera":{"yaw":5,"pitch":-3}}}`.
     The response includes `observation`, `reward`, `terminated`, `truncated`, and `info`.

  4. Get a rendered frame

     ```bash
     curl -X POST http://localhost:8000/render \
       | jq -r '.image_base64' | base64 -d > render_frame.png
     ```

     If `last_obs` is empty because `reset` has not been called, the API returns 400.

  5. Close the environment

     ```bash
     curl -X POST http://localhost:8000/close
     ```

  6. Remote save example in Python

     ```python
     import base64, requests, pathlib
     resp = requests.post("http://100.101.28.243:8000/render")
     resp.raise_for_status()
     pathlib.Path("render_frame.png").write_bytes(
         base64.b64decode(resp.json()["image_base64"])
     )
     ```

- Smoke test inside the container:

  `python -c "from env.mc.mc_env import MCGym; env=MCGym(); env.reset(); env.close()"`.

## 2. Base Image

- Use `registry.h.pjlab.org.cn/ailab-evobox-evobox_cpu/ray:base`, about 10.3 GB and 8 weeks old. The image includes the Ray runtime and targets CPU scenarios.
- Keep `ARG RAY_BASE_IMAGE` for future replacement, but lock the default value to the image above.
- Inside the image, use `/workspace/AIEvoBox` as the project root by default. `PROJECT_ROOT` can be overridden.
- GPU/CUDA dependencies are currently out of scope. The Dockerfile and run instructions are written from a CPU-only perspective.

## 3. System Dependencies

- Install the following APT packages for `opencv-python`, `pyglet`, `Pyro4`, rendering, and video components:

  `xvfb openjdk-8-jre-headless ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libxi6 libxrandr2 libxcursor1 libxinerama1 libvulkan1 libsdl2-dev mesa-utils`.

- Also install `build-essential cmake git curl python3-dev pkg-config` for dependencies that require compilation.
- Set locale, such as `en_US.UTF-8`, and timezone to avoid Ray runtime warnings.

## 4. Python Dependencies and MineStudio Installation

1. Copy `env/mc/requirements.txt` into the image, for example `/tmp/requirements_mc.txt`, and run `pip install --no-cache-dir -r /tmp/requirements_mc.txt`.
2. Use `ARG PROJECT_ROOT=/workspace/AIEvoBox`, create the directory in the image, and `COPY . ${PROJECT_ROOT}`.
3. Run `pip install --no-cache-dir -e ${PROJECT_ROOT}/env/mc/MineStudio` inside the container so editable install takes effect.
4. If `requirements.txt` adds dependencies later, update that file first instead of duplicating `pip install` lines in the Dockerfile.
5. Use `ENV PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH}` so the FastAPI service can import project modules.

## 5. Absolute Path Handling

`minestudio/simulator/minerl/env/malmo.py`, `minestudio/simulator/entry.py`, and `minestudio/utils/temp.py` now prefer `MINECRAFT_JAR_PATH` / `MINESTUDIO_DIR` and fall back to `PROJECT_ROOT`, defaulting to `/workspace/AIEvoBox`. This avoids hard-coded host paths.

- The default jar location is `${PROJECT_ROOT}/env/mc/build/offline-mcprec-6.13.jar`, and it can be overridden by environment variables.
- Keep `ENV PYTHONPATH=${PROJECT_ROOT}` so project modules can be imported.

## 6. FastAPI Service Design

- Suggested locations: `env/mc/service/app.py` and `env/mc/service/env_manager.py`.
- Core components:
  - `EnvManager`: lazy-loads `MCGym` and uses `asyncio.Lock` / `threading.Lock` to preserve a singleton.
  - `@app.post("/reset")`: returns `obs` and `info`.
  - `@app.post("/step")`: accepts action JSON and returns `obs/reward/terminated/truncated/info`.
  - `@app.post("/render")`: calls `env.render()` and returns PNG Base64 or bytes.
  - `@app.post("/close")`: releases resources.
  - `@app.get("/healthz")`: quickly checks service status and jar availability.
- For long step operations, use background tasks or synchronous blocking, keeping `uvicorn --workers 1 --host 0.0.0.0 --port 8000`. Multiple workers would require process-level isolation in `EnvManager`.

## 7. Dockerfile Structure Draft

```text
ARG RAY_BASE_IMAGE=registry.h.pjlab.org.cn/ailab-evobox-evobox_cpu/ray:base
ARG PROJECT_ROOT=/workspace/AIEvoBox
FROM ${RAY_BASE_IMAGE} AS runtime

# 1. Install system dependencies with apt, including xvfb and openjdk-8-jre-headless
# 2. Copy requirements and pip install
# 3. Create ${PROJECT_ROOT} and COPY the full repository
# 4. pip install -e ${PROJECT_ROOT}/env/mc/MineStudio
# 5. ENV PYTHONPATH=${PROJECT_ROOT}
# 6. ENTRYPOINT: uvicorn env.mc.service.app:app --host 0.0.0.0 --port 8000
```

- Optional multi-stage build: if the jar needs a separate build stage, handle it in an earlier stage.
- For faster rebuilds, use a `pip wheel` cache layer.

## 8. Items to Confirm

- ✅ Base image: `registry.h.pjlab.org.cn/ailab-evobox-evobox_cpu/ray:base`.
- ✅ `offline-mcprec-6.13.jar` is distributed with the image and kept in `AIEvoBox/env/mc/build/`.
- ✅ FastAPI does not need authentication.
- ✅ GPU/EGL dependencies are not considered for now.
- ✅ API schema is not versioned yet.
- ✅ Action and observation JSON examples are provided in the run steps.
- Still to discuss: log/model-output mount paths, Ray cluster integration, and other details.
