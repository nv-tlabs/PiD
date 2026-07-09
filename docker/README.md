# PiD NGC container

The default base is `nvcr.io/nvidia/pytorch:25.03-py3`. NVIDIA's
[25.03 release notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-03.html)
list Python 3.12, CUDA 12.8.1, PyTorch 2.7.0a0, Transformer Engine 2.1, cuDNN
9.8, and NCCL 2.25.1; Apex is included as well. The image keeps that tested NGC
framework stack intact and installs only the remaining PiD dependency graph into
NGC's system Python. It does not create a virtual environment.

The repository lock continues to pin PyTorch 2.10 and TorchVision 0.25 for local
non-NGC environments. The container instead installs the checked-in
`requirements-ngc-overlay.txt` with `--no-deps`; this shared aarch64/x86_64 file
contains 48 exact version pins and no artifact hashes. `uv` prints each change.
PyTorch, TorchVision, Transformer Engine, Apex, Triton, NVIDIA packages, and
OpenCV are not listed and therefore remain the NGC versions.

## Requirements

- Docker with BuildKit support.
- NVIDIA Container Toolkit and a compatible NVIDIA driver. CUDA 12.8 on
  Blackwell should use an R570-or-newer driver.
- Access to `nvcr.io`. If authentication is requested, run
  `docker login nvcr.io`, use `$oauthtoken` as the username, and use an NGC API
  key as the password.

The NGC PyTorch image and the shared overlay support Linux x86_64 and Arm SBSA.

## Build

Run the build from the repository root. `pid-ngc:25.03` is the local tag created
by this command; `--pull` refreshes only the NGC base image.

```bash
DOCKER_BUILDKIT=1 docker build --pull \
    --file docker/Dockerfile \
    --tag pid-ngc:25.03 \
    .
```

## Enroot

Enroot does not execute the Dockerfile. Import the same NGC base and install the
same requirements in a temporary writable rootfs:

```bash
enroot import --output docker/ngc-pytorch-25.03-py3-arm64.sqsh \
    'docker://nvcr.io#nvidia/pytorch:25.03-py3'

enroot create --name pid-ngc2503-build \
    docker/ngc-pytorch-25.03-py3-arm64.sqsh

enroot start --root --rw pid-ngc2503-build \
    mkdir -p /tmp/pid /tmp/uv-cache

enroot start --root --rw \
    --env PYTHONNOUSERSITE=1 \
    --env UV_CACHE_DIR=/tmp/uv-cache \
    --mount "$(pwd):/tmp/pid:none:bind,ro" \
    pid-ngc2503-build \
    bash -lc 'PIP_CONSTRAINT= python3 -m pip install \
      --break-system-packages --no-cache-dir uv==0.11.28 && \
      /usr/local/bin/uv pip install \
      --system --break-system-packages \
      --python /usr/bin/python3.12 --no-managed-python --no-deps \
      --requirements /tmp/pid/docker/requirements-ngc-overlay.txt \
      --compile-bytecode'

enroot export --force --output docker/pid_docker_arm64.sqsh \
    pid-ngc2503-build
enroot remove --force pid-ngc2503-build
```

Start the final image with the repository mounted at `/workspace/pid`:

```bash
enroot start --root \
    --mount "$(pwd):/workspace/pid:none:bind,rw" \
    docker/pid_docker_arm64.sqsh bash
```

The first SquashFS is only the NGC rebuild cache; `pid_docker_arm64.sqsh` is the
final image. Enroot's NVIDIA hook exposes the GPUs automatically.

## Verify the Docker image

GPU access is unavailable during `docker build`, so verify CUDA after the image
has been built:

```bash
docker run --rm --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --mount type=bind,source="$(pwd)",target=/workspace/pid \
    --workdir /workspace/pid \
    pid-ngc:25.03 \
    bash -c 'PYTHONPATH=. python verify_env.py'
```

`verify_env.py` checks the required imports, CUDA visibility, and a CUDA kernel.

## Start an interactive container

Create persistent output and cache directories in the mounted repository:

```bash
mkdir -p \
    imaginaire4/imaginaire4-output \
    imaginaire4/imaginaire4-cache
```

Then start the container:

```bash
docker run --rm -it --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --mount type=bind,source="$(pwd)",target=/workspace/pid \
    --workdir /workspace/pid \
    --env IMAGINAIRE_OUTPUT_ROOT=/workspace/pid/imaginaire4/imaginaire4-output \
    --env IMAGINAIRE_CACHE_DIR=/workspace/pid/imaginaire4/imaginaire4-cache \
    --env HF_HOME=/workspace/pid/imaginaire4/imaginaire4-cache/huggingface \
    --env TORCH_HOME=/workspace/pid/imaginaire4/imaginaire4-cache/torch \
    --env WANDB_CACHE_DIR=/workspace/pid/imaginaire4/imaginaire4-cache/wandb/cache \
    --env WANDB_DATA_DIR=/workspace/pid/imaginaire4/imaginaire4-cache/wandb/data \
    --env WANDB_API_KEY \
    --env HF_TOKEN \
    pid-ngc:25.03
```

`WANDB_API_KEY` and `HF_TOKEN` are forwarded from the host environment. Omit
either flag when it is not needed. To mount a dataset outside the repository,
add, for example:

```bash
--mount type=bind,source=/path/on/host,target=/data,readonly
```

By default the container runs as root. To avoid root-owned output files, add the
following flags; the installed environment is readable by an arbitrary user:

```bash
--user "$(id -u):$(id -g)" --env HOME=/tmp
```

## Shared memory

PyTorch dataloaders and NCCL need more than Docker's default 64 MB shared-memory
allocation. The examples use `--ipc=host`, following NVIDIA's
[container guidance](https://docs.nvidia.com/deeplearning/frameworks/user-guide/index.html#setting-the-shared-memory-flag).
If sharing the host IPC namespace is not allowed, replace it with an explicit
allocation such as:

```bash
--shm-size=64g
```

Do not pass `--ipc=host` and `--shm-size` together.
