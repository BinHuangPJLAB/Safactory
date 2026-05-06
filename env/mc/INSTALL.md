# MCGym Environment Installation Guide

## 📦 Installation Options

### Option 1: Use requirements.txt (recommended)

```bash
cd /fs-computility-new/nuclear/leishanzhe/repo/AIEvoBox/env/mc
pip install -r requirements.txt
```

### Option 2: Install MineStudio in development mode

```bash
cd /fs-computility-new/nuclear/leishanzhe/repo/AIEvoBox/env/mc/MineStudio
pip install -e .
```

### Option 3: Install only key dependencies

```bash
# Minimal install
pip install minecraft-data rich pyyaml pillow numpy

# Full functionality
pip install -r requirements.txt
```

## 🔧 Dependency Notes

### Core Dependencies

- **minecraft-data** - Minecraft data access library, required
- **rich** - Pretty console output
- **pyyaml** - YAML config parsing
- **pillow** - Image processing
- **numpy** - Numerical computing

### MineStudio Dependencies

The full MineStudio dependency list is in `requirements.txt`, including:

- PyTorch and related libraries
- Gymnasium/Gym
- Various callbacks and utility libraries

## ✅ Verify Installation

### Quick Test

```bash
cd /fs-computility-new/nuclear/leishanzhe/repo/AIEvoBox
python env/mc/test/test_mc_env_simple.py
```

The expected output should show most tests passing.

### Full Test

```bash
# After installing all dependencies
python env/mc/test/test_mc_env.py
```

### Python Check

```python
# Test basic imports
import minecraft_data
from env.mc.mc_env import MCGym
print("✓ Installed successfully!")
```

## ⚠️ Common Issues

### 0. `xvfb-run` command not found

**Issue**: Running the environment reports:

```text
xvfb-run: command not found
```

or the full error:

```text
/fs-computility-new/nuclear/leishanzhe/repo/AIEvoBox/env/mc/MineStudio/minestudio/simulator/minerl/env/launchClient.sh: line 38: xvfb-run: command not found
```

**Cause**:

- Xvfb (X Virtual Framebuffer) is missing. It is a required system dependency for running the Minecraft GUI in a headless server environment.

**Solution**:

Install xvfb according to your Linux distribution:

**Ubuntu/Debian:**

```bash
sudo apt-get update
sudo apt-get install -y xvfb
```

**CentOS/RHEL:**

```bash
sudo yum install -y xorg-x11-server-Xvfb
```

**Fedora:**

```bash
sudo dnf install -y xorg-x11-server-Xvfb
```

**Verify installation:**

```bash
which xvfb-run
# Expected output: /usr/bin/xvfb-run
```

Run your script again after installation.

### 1. Environment not registered

**Issue**: Running `base_eval.py` reports:

```text
Program failed: environment 'mc_gym' is not registered. Please register it with the @register_env decorator first.
```

**Cause**:

- The Python decorator `@register_env` only runs when the module is imported.
- `base_eval.py` did not import the `MCGym` class, so the decorator never ran and the environment was not registered.

**Solution**:

Add this import in `AIEvoBox/examples/base_eval.py` (**already fixed**):

```python
from env.mc.mc_env import MCGym  # Import the Minecraft environment for registration
```

**Verify**:

```bash
cd /fs-computility-new/nuclear/leishanzhe/repo/AIEvoBox
bash examples/run_1_mc_envs.sh
```

After running, `mc_gym` should appear in the registered environment list:

```text
Registered environment types: ['android_gym', 'core_git_env', 'embodied_alfred', 'mc_gym']
```

---

### 2. `ModuleNotFoundError: No module named 'minecraft_data'`

**Issue**: The `minecraft_data` module is missing.

**Solution**:

```bash
pip install minecraft-data
```

### 3. gym installation failure / import gym error

**Issue**: `ModuleNotFoundError: No module named 'gym'`

MineStudio internal code uses `import gym`, but gym is no longer maintained and is incompatible with newer setuptools versions.

**Solution**:

Use gymnasium, the maintained version of gym, plus a compatibility layer:

```bash
# Install gymnasium and compatibility layer
pip install gymnasium>=0.26.0 shimmy[gym-v21]>=0.2.1
```

`mc_env.py` already handles compatibility by mapping gymnasium to the gym module.

### 4. CUDA runtime error (`gpu_utils.py`)

**Issue**: `ImportError: cannot import name 'cuda' from 'cuda'`

This is a GPU-related issue when MineStudio runs the Minecraft simulator.

**Cause**:

- The `cuda-python` package is not installed correctly.
- Or the system has no available CUDA environment.

**Solution**:

#### Option A: Install CUDA support (recommended when an NVIDIA GPU is available)

```bash
# 1. Make sure CUDA Toolkit 11.x or 12.x is installed
nvidia-smi  # Check CUDA version

# 2. Install the matching cuda-python package
pip install cuda-python

# 3. Verify installation
python -c "from cuda import cuda, cudart; print('CUDA OK')"
```

#### Option B: Run in CPU mode (when no GPU is available)

MineStudio currently requires GPU by default. CPU mode requires modifying the MineStudio config or using simulated tests.

**Test result notes**:

- ✓ Environment initialization succeeded: all Python dependencies are resolved.
- ✗ Simulator execution failed: CUDA environment still needs to be fixed.

### 5. Java JAXB error at Minecraft runtime

**Issue**: `java.lang.NoClassDefFoundError: javax/xml/bind/JAXBException`

**Cause**:

- JAXB (Java XML Binding) was removed in Java 9+.
- Minecraft Malmo requires JAXB to parse XML configuration.

**Solution: use Java 8 (recommended)**

```bash
# 1. Check current Java version
java -version

# 2. Install Java 8
sudo apt-get install openjdk-8-jdk

# 3. Set Java 8 as default
sudo update-alternatives --config java
# Select the java-8 option

# 4. Verify
java -version  # Should show 1.8.x

# Or temporarily use Java 8
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH
```

**Alternative: add JAXB in Java 11+**

If Java 11+ must be used, add the JAXB module when launching:

```bash
# Modify MineStudio's launchClient.sh
# Add the --add-modules java.xml.bind argument
```

### 6. Action format conversion issues

**Issue A**: `TypeError: string indices must be integers` at `action["buttons"]`

**Cause**:

- `MCGym.step(action)` receives an action in string format from the LLM.
- `MinecraftSim.step()` expects a dictionary action with `buttons` and `camera` keys.

**Issue B**: `AttributeError: 'int' object has no attribute 'ndim'`

**Cause**:

- `MinecraftSim` internally expects `action["camera"]` to be a numpy array in `action_mapper.to_factored()`.
- `ActionFromLLMConverter` returns int values by default.

**Issue C**: `IndexError: index 220 is out of bounds for axis 0 with size 121`

**Cause**:

- `ActionFromLLMConverter` uses **21×21=441** camera bins for LLM output.
- `MinecraftSim.action_mapper` expects **11×11=121** camera bins in VPT format.
- The camera index mismatch causes an out-of-bounds error.

**Solution**:

`mc_env.py` already configures the action converter correctly:

```python
# Initialize converter in __init__
self.action_converter = ActionFromLLMConverter(
    hfov_deg=self.current_hfov,
    vfov_deg=self.current_vfov,
    return_numpy=True,      # Return numpy array format
    map_camera_to_11=True   # Map 21×21 to 11×11 (VPT)
)

# Automatically convert string actions in step()
if isinstance(action, str):
    image_shape = self.obs_size  # Read from config
    action_dict = self.action_converter.convert(action, image_shape)
else:
    action_dict = action
```

### 7. MinecraftSim return type issues

**Issue**: `AttributeError: 'tuple' object has no attribute 'obs'`

**Cause**:

- `MinecraftSim.reset()` and `MinecraftSim.step()` may return either tuples or objects.
- Different MineStudio versions may use different return formats.

**Solution**:

`mc_env.py` already handles this compatibility:

```python
# reset() handles tuple or object
if isinstance(result, tuple):
    obs, info = result if len(result) == 2 else (result[0], {})
else:
    obs = result.obs if hasattr(result, 'obs') else result
    info = result.info if hasattr(result, 'info') else {}

# step() handles tuples of different lengths
if isinstance(result, tuple):
    # Supports (obs, reward, done, info) or (obs, reward, terminated, truncated, info)
    if len(result) == 4:
        obs, reward, done, info = result
        terminated = done
        truncated = False
    elif len(result) == 5:
        obs, reward, terminated, truncated, info = result
```

### 8. Minecraft runtime warnings that can be ignored

The following errors and warnings are **normal** and do not affect simulator functionality:

#### ✅ Messages that can be ignored

```text
ERROR : Couldn't load Narrator library
```

- **Description**: Text narration is not needed by the simulator.

```text
Error starting SoundSystem
Failed to open OpenAL device
```

- **Description**: Headless servers have no audio device; this is normal.

```text
Unable to load minecraft:optifine/ctm/default/empty.png
Missing CTM sprite
```

- **Description**: Optional OptiFine texture; does not affect functionality.

```text
Could not authorize you against Realms server
```

- **Description**: Minecraft Realms online features are not needed by the simulator.

```text
Ambiguity between arguments [teleport, ...]
```

- **Description**: Warning from the Minecraft command parser; does not affect functionality.

#### ✅ Successful startup signal

Minecraft has started successfully when you see:

```text
INFO - Minecraft process ready
INFO - Logging output of Minecraft to ./logs/mc_...log
```

### 9. cuda-python installation failure

**Issue**: `pip install cuda-python` fails.

**Solution**:

```bash
# Skip cuda-python; other dependencies can still work
pip install -r requirements.txt --ignore-installed cuda-python || true
```

### 10. torch version conflict

**Issue**: PyTorch version incompatibility.

**Solution**:

```bash
# Install PyTorch first, matching your CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Then install other dependencies
pip install -r requirements.txt
```

### 11. pyrender installation failure

**Issue**: `pyrender==0.1.25` has complex dependencies.

**Solution**:

```bash
# Try a newer version or skip dependency resolution
pip install pyrender  # Install latest version
# Or
pip install -r requirements.txt --no-deps
```

## 📝 Dependency List

Main dependencies include:

```text
minecraft-data>=0.7.0
av
opencv-python
numpy
torch>=2.3.1
gymnasium
rich
pyyaml
pillow
ray
transformers
hydra-core>=1.3.2
... (see requirements.txt for the full list)
```

## 🎯 Minimal Install for Config Tests

If you only want to test config and basic functionality without full MinecraftSim:

```bash
pip install pyyaml rich pillow numpy pydantic
```

Then run:

```bash
python env/mc/test/test_mc_env_simple.py
```

## 📚 More Information

- MineStudio official documentation: see local `MineStudio/README.md` if MineStudio has been pulled.
- Test guide: see local `test/README.md` if the test directory is available.
- Configuration notes: see local `mc_env.yaml` if that config file exists.
