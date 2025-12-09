FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# minimal OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# install Python deps (CPU wheels for torch/torchvision)
COPY requirements-docker.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install -r /tmp/requirements.txt

# copy project
COPY . /app
ENV PYTHONPATH=/app

# default command just shows help
CMD ["python", "-m", "src.ci.drift_gate", "--help"]
