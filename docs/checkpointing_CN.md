# PiD 训练 Checkpointing 与 DCP 格式

## 概述

本项目训练时主要使用 PyTorch Distributed Checkpoint（DCP）保存模型和训练状态。DCP 适合分布式训练的保存与续训；推理或新阶段权重初始化通常更适合使用单文件 `.pth`。

本文介绍：

1. DCP 格式是什么；
2. teacher training 与 distillation training 的真实 checkpoint 目录结构；
3. 如何使用 `scripts/convert_distcp_to_pt.py` 将 DCP 模型权重合并为单文件 `.pth`。

发布模型的路径索引仍见 [checkpoints.md](checkpoints.md)。本文只讨论训练过程中产生的 checkpoint。

## 关键文件

| 文件 | 作用 |
|---|---|
| `pid/_ext/imaginaire/checkpointer/dcp.py` | 普通 teacher/model 训练使用的 `DistributedCheckpointer` |
| `pid/_src/checkpointer/dcp_distill.py` | 多 optimizer/scheduler 蒸馏训练使用的 `DistillationCheckpointer` |
| `pid/_src/configs/common/defaults/ckpt_type.py` | 注册 Hydra 的 `dcp` 与 `dcp_distill` 选项 |
| `scripts/convert_distcp_to_pt.py` | 将一个 DCP state-dict 目录转换为单文件 `.pth` |

Teacher 实验通常选择：

```python
{"override /ckpt_type": "dcp"}
```

Distillation 实验通常选择：

```python
{"override /ckpt_type": "dcp_distill"}
```

## 什么是 DCP 格式

DCP 是 `torch.distributed.checkpoint` 提供的目录式、分片 checkpoint 格式。一个逻辑 state dict 不是保存成单个 `.pt`/`.pth` 文件，而是保存为一个目录：

```text
model/
├── .metadata
├── __0_0.distcp
├── __1_0.distcp
└── ...
```

- `.metadata`：记录 state-dict key、tensor 的 shape/dtype/chunk，以及这些数据在 shard 文件中的位置；
- `__<rank>_<index>.distcp`：保存 tensor 和非 tensor 的实际 payload；
- 整个目录共同构成一个 DCP checkpoint，不能只拿其中某个 `.distcp` 文件使用，也不应手工修改 `.metadata`。

DCP 的主要优点是多进程并行 I/O、降低单个进程保存/加载时的内存压力，并允许 DCP loader 按当前分布式 state dict 读取所需 shard。代价是它不是可以直接 `torch.load()` 的单文件格式，浏览、迁移和推理使用都不如 `.pth` 直接。

本项目保存时使用：

```python
DefaultSavePlanner(dedup_save_to_lowest_rank=True)
```

对于 DDP 中各 rank 完全复制的状态，payload 可能只写入最低 rank，其他 rank 对应的 `.distcp` 文件可能为 0 bytes。这在本文的两个样例中都存在，**不代表 checkpoint 损坏**。在 FSDP/DTensor 等真正分片的场景下，数据分布可能不同，因此 shard 数量和大小都不是固定格式。

## 通用 checkpoint 层级

每次保存的 iteration 不足九位时会在左侧补零：

```text
checkpoints/
├── latest_checkpoint.txt
├── iter_000000025/
├── iter_000005000/
└── ...
```

`latest_checkpoint.txt` 不是软链接，其内容是最近一次成功保存的目录名，例如：

```text
iter_000000025
```

同一个 job 目录重新启动时，checkpointer 会优先读取该文件；默认从对应 iteration 恢复 model、optim、scheduler 和 trainer，也可以通过 `keys_not_to_resume` 排除指定部分。

## Teacher training checkpoint

本文核对的 teacher 样例位于：

```text
imaginaire4/imaginaire4-interactive-output/pid_training/pid_training_v1pt5_debug/
pid_v1pt5_teacher_flux2_h1024_d4_fix_backbone_res_2048_to_3840_W_RESUME_2026-07-08_08-58-07/checkpoints
```

实际目录结构为：

```text
checkpoints/
├── latest_checkpoint.txt                 # iter_000000025
└── iter_000000025/
    ├── model/
    │   ├── .metadata
    │   ├── __0_0.distcp
    │   └── __1_0.distcp
    ├── optim/
    │   ├── .metadata
    │   ├── __0_0.distcp
    │   └── __1_0.distcp
    ├── scheduler/
    │   ├── .metadata
    │   ├── __0_0.distcp
    │   └── __1_0.distcp
    └── trainer/
        ├── .metadata
        ├── __0_0.distcp
        └── __1_0.distcp
```

各目录含义：

| 目录 | 内容 |
|---|---|
| `model/` | 模型普通权重和 EMA 权重 |
| `optim/` | 单个 optimizer 的状态，例如 AdamW 的一阶矩/二阶矩状态 |
| `scheduler/` | 单个学习率 scheduler 的状态 |
| `trainer/` | iteration 和 GradScaler 状态 |

该样例的 metadata 实际包含：

- `model/`：934 个 key，其中 `net.*` 467 个、`net_ema.*` 467 个；
- `optim/`：单个 optimizer 的 `state.*` 与 `param_groups.*`；
- `scheduler/`：`base_lrs`、`last_epoch`、`_step_count` 等字段；
- `trainer/`：本次 debug run 配置了 `trainer.grad_scaler_args.enabled=False`，所以 metadata 里只有 `iteration` 这个 key；从 shard payload 加载出的实际值为 25。

代码保存时仍会同时提供 `grad_scaler.state_dict()` 和 `iteration`；其他精度或配置下，`trainer/` 可能包含更多字段，不应假设它永远只有 iteration。

## Distillation training checkpoint

本文核对的 distillation 样例位于：

```text
imaginaire4/imaginaire4-interactive-output/pid_training/pid_training_v1pt5_debug/
pid_v1pt5_student_flux_h1024_d4_res_2048_distill_W_RESUME_2026-07-08_08-35-44/checkpoints
```

实际目录结构为：

```text
checkpoints/
├── latest_checkpoint.txt                 # iter_000000025
└── iter_000000025/
    ├── model/                             # model state
    ├── optim_net/                         # student optimizer
    ├── optim_fake_score/                  # fake-score optimizer
    ├── optim_discriminator/               # discriminator optimizer
    ├── scheduler_net/                     # student scheduler
    ├── scheduler_fake_score/              # fake-score scheduler
    ├── scheduler_discriminator/           # discriminator scheduler
    └── trainer/                           # iteration / GradScaler
```

上面的每个目录内部都有自己的 `.metadata` 和 `__0_0.distcp` 到 `__3_0.distcp`。这些 optimizer/scheduler 被拆成独立 DCP 目录，以便分别维护各自的参数映射。

目录后缀来自模型的 `optimizer_dict` / `scheduler_dict` key，规则是：

```text
optim_<key>/
scheduler_<key>/
```

因此这些名字取决于当前配置和启用的组件，并不是永远固定为 `net`、`fake_score`、`discriminator`。例如关闭 GAN/discriminator 后，相应的 model key、optimizer 和 scheduler 目录可能不存在。

该样例 `model/.metadata` 实际包含 1389 个 key：

| 前缀 | 数量 | 含义 |
|---|---:|---|
| `net.*` | 461 | student 普通权重 |
| `net_ema.*` | 461 | student EMA 权重 |
| `fake_score.*` | 461 | fake-score 网络权重 |
| `discriminator.*` | 6 | discriminator 权重 |

### Frozen teacher 不在 distillation DCP 中

`PidDistillModel.state_dict()` 不保存 frozen teacher。teacher 由配置中的 `pretrained_teacher_path` 单独加载，因此 distillation DCP 并不是 teacher 与 student 的完整打包文件。

迁移或恢复蒸馏训练环境时，除了 DCP 目录，还必须保证配置引用的 teacher checkpoint 可访问且兼容。

## Teacher 与 distillation 结构对比

| 项目 | Teacher training | Distillation training |
|---|---|---|
| Hydra `ckpt_type` | `dcp` | `dcp_distill` |
| Checkpointer | `DistributedCheckpointer` | `DistillationCheckpointer` |
| Model keys | `net.*`、`net_ema.*` | `net.*`、`net_ema.*`、`fake_score.*`、可选 `discriminator.*` |
| Optimizer | 单个 `optim/` | 每个组件一个 `optim_<key>/` |
| Scheduler | 单个 `scheduler/` | 每个组件一个 `scheduler_<key>/` |
| Trainer state | `trainer/` | `trainer/` |
| Frozen teacher | 不适用 | 不保存在 DCP 中，需单独提供 |

样例中的 teacher 每个逻辑目录有 2 个 `.distcp` 文件，distillation 有 4 个，是因为两个运行使用的分布式规模不同，而不是两种格式规定了固定 shard 数量。

## 将 DCP 转换为单文件 `.pth`

使用：

```text
scripts/convert_distcp_to_pt.py
```

转换模型时，输入必须是本地文件系统中的 DCP 目录，并精确指向某个 iteration 下的 `model/`：

```text
.../checkpoints/iter_000000025/model
```

不要传入 `checkpoints/` 根目录，也不要只传 `iter_000000025/`。该脚本不直接接受 `s3://...` 等对象存储 URI；需要先把 DCP 完整下载到本地。

### 基本转换

```bash
mkdir -p /path/to/output_dir
PYTHONPATH=. python scripts/convert_distcp_to_pt.py \
  /path/to/checkpoints/iter_000000025/model \
  /path/to/output_dir
```

默认生成：

```text
/path/to/output_dir/model.pth
```

`model.pth` 是传入的 `model/` DCP state dict 的完整合并结果。由于 optimizer、scheduler 和 trainer 分别保存在其他目录，这个文件不包含它们。

### 导出 EMA 推理权重

```bash
mkdir -p /path/to/output_dir
PYTHONPATH=. python scripts/convert_distcp_to_pt.py \
  /path/to/checkpoints/iter_000000025/model \
  /path/to/output_dir \
  --ema
```

`--ema` 会生成三个文件：

| 输出文件 | 内容 |
|---|---|
| `model.pth` | `model/` 中的完整 state dict |
| `model_ema_fp32.pth` | 仅保留 `net_ema.*` 并重命名为 `net.*`；脚本保留源 dtype，本文两个样例恰好为 fp32 |
| `model_ema_bf16.pth` | 在上述 EMA-only 结果上，将原 dtype 为 float32 的 tensor 转为 bfloat16，其他 dtype/对象保持不变 |

对于 distillation checkpoint，`--ema` 导出的是 **student EMA**；`fake_score.*` 和 `discriminator.*` 不会进入 EMA-only 文件，frozen teacher 本来就不在该 DCP 中。

如果 checkpoint 没有 `net_ema.*`，脚本会先写出 `model.pth`，随后在 EMA 提取阶段报错。

### 使用本文 teacher 样例转换

从仓库根目录运行：

```bash
INPUT="imaginaire4/imaginaire4-interactive-output/pid_training/pid_training_v1pt5_debug/pid_v1pt5_teacher_flux2_h1024_d4_fix_backbone_res_2048_to_3840_W_RESUME_2026-07-08_08-58-07/checkpoints/iter_000000025/model"
OUTPUT="checkpoints/converted/teacher_flux2_iter_000000025"

mkdir -p "$OUTPUT"
PYTHONPATH=. python scripts/convert_distcp_to_pt.py "$INPUT" "$OUTPUT" --ema
```

Distillation 样例的转换方式相同，只需将 `INPUT` 改为其 `iter_000000025/model` 目录。

## 转换与恢复训练的注意事项

- 使用普通 `python` 单进程执行转换，不要用 `torchrun`；转换会把整个逻辑 state dict 合并进 CPU 内存。
- 脚本不会创建输出目录，必须先执行 `mkdir -p`。
- 脚本启动后会先删除输出目录中已有的 `model.pth`、`model_ema_fp32.pth` 和 `model_ema_bf16.pth`，然后才开始转换；即使不传 `--ema`，或随后因输入错误/OOM 失败，旧输出也已被删除。转换前务必确认输出路径。
- 本文样例的 `model/` payload 约为 teacher 11 GB、distillation 16 GB，转换时需要预留足够的 CPU RAM 和额外磁盘空间，峰值内存可能明显高于文件大小。
- `--keep-original` 会在写出 `model.pth` 后直接退出，不提取 EMA。默认不加 `--ema` 时也只生成 `model.pth`；若同时传 `--keep-original --ema`，`--keep-original` 优先，不会生成 EMA-only 文件。
- 脚本实际生成的扩展名是 `.pth`；`.pt` 和 `.pth` 都常用于 PyTorch 单文件 checkpoint，但不要按 `.pt` 名称寻找本脚本的输出。
- 转换 `model/` 得到的 `.pth` 适合推理或新阶段权重初始化，但不包含 optimizer、scheduler、trainer iteration，**不能替代完整 DCP 做无损续训**。

### DCP 恢复路径

- **同一 job 自动续训**：如果当前 job 的 `checkpoints/latest_checkpoint.txt` 存在，checkpointer 会优先加载它指向的本地 DCP，默认严格恢复 model、optimizer、scheduler 和 trainer；`load_training_state=False` 不会关闭这种 self-resume，只有 `keys_not_to_resume` 可以排除指定部分。
- **显式加载外部 DCP**：将 `checkpoint.load_path` 指向外部的 `iter_xxxxxxxxx/`。`load_training_state=False` 时通常只加载 `model/`；设为 `True` 才同时加载 optimizer、scheduler 和 trainer。这种方式不依赖外部目录的 `latest_checkpoint.txt`。
- **加载转换后的 `.pth`**：只用于模型权重初始化，从 iteration 0 开始，不能恢复 optimizer、scheduler 或原 trainer iteration。

因此，从同一 job 的中断位置自动完整恢复时，应保留 `checkpoints/iter_xxxxxxxxx/` 目录以及同级的 `checkpoints/latest_checkpoint.txt`，并让项目 checkpointer 直接加载 DCP。
