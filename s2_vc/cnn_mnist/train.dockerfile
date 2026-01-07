## 0. Generic Details

# Base image
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

#  more or less a Python installation and build tools
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# 1. Project-specific commands

# essential components of project given current cookiecutter setup
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY uv.lock uv.lock
COPY src/ src/
COPY data/ data/

# set working dir 
WORKDIR /

# install dependencies (uv specific, uses cache)
ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv uv sync

# # for pip install, run: 
# RUN pip install -r requirements.txt --no-cache-dir
# RUN pip install . --no-deps --no-cache-dir

# Name our training script as the entrypoint for our Docker image. (uv specifc)
# since his is the application that we want to run when the image is executed:
ENTRYPOINT ["uv", "run", "src/cnn_mnist/train.py"]


