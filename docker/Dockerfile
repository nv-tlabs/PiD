# syntax=docker/dockerfile:1.7

# Keep the CUDA, PyTorch, TorchVision, Transformer Engine, and Apex stack from
# NGC as one ABI-compatible unit. This release supports x86_64 and Arm SBSA.
FROM nvcr.io/nvidia/pytorch:25.03-py3

ARG UV_VERSION=0.11.28

SHELL ["/bin/bash", "-euxo", "pipefail", "-c"]

# Install the PiD dependency overlay into NGC's system Python. In particular,
# do not put a virtual environment ahead of NGC's optimized Python packages.
ENV UV_LINK_MODE=copy \
    UV_NO_PROGRESS=1 \
    PYTHONNOUSERSITE=1 \
    PYTHONUNBUFFERED=1

# Install only the explicitly pinned overlay packages. --no-deps keeps uv from
# resolving or replacing the NGC framework/CUDA stack.
RUN --mount=type=bind,target=/tmp/pid,readonly \
    --mount=type=cache,target=/root/.cache/uv \
    PIP_CONSTRAINT= python3 -m pip install \
        --break-system-packages --no-cache-dir "uv==${UV_VERSION}" && \
    /usr/local/bin/uv pip install \
        --system \
        --break-system-packages \
        --python /usr/bin/python3.12 \
        --no-managed-python \
        --no-deps \
        --requirements /tmp/pid/docker/requirements-ngc-overlay.txt \
        --compile-bytecode

WORKDIR /workspace/pid

# Keep the NGC entrypoint; it performs NVIDIA container initialization.
CMD ["/bin/bash"]
