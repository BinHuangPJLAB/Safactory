# Environment Deployment

Two deployment modes are provided:

- Local deployment
- Docker deployment

## Local Deployment

```bash
# 1. Create a conda environment
conda create -n qa-gym python=3.10 -y
conda activate qa-gym

# 2. Pin `starlette` for FastAPI compatibility:
echo "starlette>=0.40.0,<0.49.0" > constraints.txt

# 3. Install dependencies
pip install -r requirements.txt -c constraints.txt
pip install rayjob_sdk-0.3.11-py3-none-any.whl -c constraints.txt
pip install "protobuf>=6,<7" openrt -c constraints.txt

# 4. Configure environment variables
export OPENAI_API_KEY=your_key
export OPENAI_BASE_URL=your_url
```

## Docker Deployment

- Build the image with the Dockerfile. The OPENAI environment variables must be configured.

```bash
docker build -f ./env/qagym/Dockerfile -t qa-gym:1.3 .
```

- Pull the image directly:

```bash
docker pull registry.h.pjlab.org.cn/ailab-evobox-evobox_cpu/qa-gym:1.3
```
