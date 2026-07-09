# PiD 训练配置系统简明教程

## 概述

本项目的训练配置不是常见的 YAML 目录加 `@hydra.main`，而是一套基于以下组件的 Python 配置系统：

- `attrs`：定义有类型和默认值的根配置；
- Hydra `ConfigStore`：注册模型、数据、优化器和实验等配置组；
- Hydra `compose()` 与 OmegaConf：按 `defaults` 和命令行 override 合成最终配置；
- `LazyCall`：记录对象的构造函数和参数，在训练开始时才真正实例化。

训练配置入口是 `pid/_src/configs/pid_training/config.py`，本文以
`pixeldit_text_to_image_finetune_res_2048` 为例说明整个流程。

## 快速开始

从仓库根目录启动 4 卡训练：

```bash
PYTHONPATH=. torchrun --nproc_per_node=4 --master_port=12341 -m scripts.train \
  --config=pid/_src/configs/pid_training/config.py \
  -- experiment="pixeldit_text_to_image_finetune_res_2048"
```

这条命令可以拆成三部分：

1. `torchrun ... -m scripts.train` 启动分布式训练入口；
2. `--config=.../config.py` 指定 Python 根配置模块；
3. 独立的 `--` 后面全部是 Hydra override，`experiment=...` 表示选择一个实验配置。

这里的独立 `--` **不能省略**。本项目的配置封装会显式检查它，以区分训练脚本参数和 Hydra 参数。实验名不含空格时，引号可以省略。


## 配置加载流程

配置从命令行到训练对象的流程如下：

```text
scripts/train.py
  -> 导入 --config 指定的模块并调用 make_config()
  -> 注册基础配置组，并导入实验模块完成实验注册
  -> hydra.compose() 处理 defaults 和命令行 override
  -> OmegaConf.resolve() 解析 ${...} 插值
  -> 将结果还原为 attrs Config
  -> LazyCall 在训练阶段实例化模型和 dataloader
```

相关实现位置：

| 文件 | 作用 |
|---|---|
| `pid/_src/configs/pid_training/config.py` | 根配置、默认组选取、组件及实验注册入口 |
| `pid/_src/configs/pid_training/defaults/` | PixelDiT/PiD 专用的模型、数据和 callback 配置 |
| `pid/_src/configs/common/defaults/` | 优化器、scheduler、checkpoint 等通用配置 |
| `pid/_src/configs/pid_training/experiment_pixeldit_finetune/finetune.py` | PixelDiT finetune 实验配置及注册 |
| `pid/_ext/imaginaire/utils/config_helper.py` | Hydra compose、override 和 attrs 配置还原 |
| `scripts/train.py` | 参数解析、配置加载和训练启动 |

## 根配置与 `defaults`

入口中的 `Config` 继承自项目通用配置，并额外声明 Hydra 的 `defaults`：

```python
defaults = [
    "_self_",
    {"data_train": "mock_image"},
    {"data_val": "mock_image"},
    {"optimizer": "adamw"},
    {"scheduler": "lambdalinear"},
    {"model": "ddp_pixeldit"},
    {"callbacks": "basic"},
    {"net": None},
    {"conditioner": None},
    {"ema": "power"},
    {"tokenizer": None},
    {"checkpoint": "local"},
    {"ckpt_type": "dummy"},
    {"experiment": None},
]
```

每一项表示“配置组名 -> 当前选项名”。例如 `optimizer: adamw` 表示从 `optimizer` 组选择名为 `adamw` 的配置。`None` 表示先预留这个组，但暂时不合入具体配置；因此命令行可以用 `experiment=<name>` 选择实验。

`_self_` 代表当前配置节点本身。它在根配置中位于最前面，所以后续基础组可以覆盖根配置中的默认值。

## 配置组、选项名与最终路径

组件通过 `ConfigStore` 注册。下面是当前入口的主要组及其最终落点：

| 配置组 | 示例选项 | 合入最终配置的位置 |
|---|---|---|
| `data_train` | `mock_image` | `dataloader_train` |
| `data_val` | `mock_image` | `dataloader_val` |
| `optimizer` | `adamw` | `optimizer` |
| `scheduler` | `lambdalinear` | `scheduler` |
| `model` | `ddp_pixeldit` | `_global_`，可同时设置 `model` 和 `trainer` |
| `callbacks` | `basic`、`wandb` | `trainer.callbacks` |
| `net` | `pixeldit_h1536_d14p2` | `model.config.net` |
| `conditioner` | `pixeldit_caption` | `model.config.conditioner` |
| `ema` | `power` | `model.config.ema` |
| `tokenizer` | `flux_vae_tokenizer` | `model.config.tokenizer` |
| `checkpoint` | `local` | `checkpoint` |
| `ckpt_type` | `dcp` | `checkpoint.type` |
| `experiment` | `pixeldit_text_to_image_finetune_res_2048` | `_global_` |

组名负责“选择哪一份配置”，`package` 决定“配置合入哪里”，两者不一定相同。例如 PixelDiT 网络注册为：

```python
cs.store(
    group="net",
    name="pixeldit_h1536_d14p2",
    package="model.config.net",
    node=PIXELDIT_H1536_D14P2,
)
```

所以选择网络时使用 `net=...`，覆盖网络字段时则使用最终路径，例如 `model.config.net.rope_mode=original`。

## 实验配置如何组合

2K PixelDiT finetune 实验的核心结构如下：

```python
PIXELDIT_TEXT_TO_IMAGE_FINETUNE_RES_2048 = LazyDict(
    dict(
        defaults=[
            {"override /data_train": "pixeldit_MultiAspect_4K_1M_1bs_2048"},
            {"override /model": "ddp_pixeldit"},
            {"override /net": "pixeldit_h1536_d14p2"},
            {"override /conditioner": "pixeldit_caption"},
            {"override /ckpt_type": "dcp"},
            {"override /optimizer": "adamw"},
            {"override /callbacks": ["basic", "wandb"]},
            {"override /checkpoint": "local"},
            {"override /tokenizer": None},
            "_self_",
        ],
        job=dict(
            group="pixeldit_finetune",
            name="pixeldit_text_to_image_finetune_res_2048",
        ),
        optimizer=dict(lr=1e-5, weight_decay=0.0),
        # model、scheduler、checkpoint、trainer 等具体参数……
    )
)
```

这里有三个关键点：

- `/data_train` 中的 `/` 表示从组合根查找该配置组；
- `override` 表示替换根 `defaults` 中已经存在的组选项，而不是再添加一个同名组；
- 实验里的 `_self_` 位于最后，因此先装配模型、数据等组件，再由实验正文覆盖它们的字段。例如 `adamw` 默认学习率是 `1e-4`，该实验最终将其改为 `1e-5`。

实验最后注册到 Hydra：

```python
cs.store(
    group="experiment",
    package="_global_",
    name="pixeldit_text_to_image_finetune_res_2048",
    node=PIXELDIT_TEXT_TO_IMAGE_FINETUNE_RES_2048,
)
```

`package="_global_"` 使实验内容直接合入根配置。因此最终字段是 `trainer.max_iter`、`model.config.image_size` 等，而不是 `experiment.trainer.max_iter`。

### 实验继承

2K 到 4K 的实验直接继承 2K 实验，再替换数据组和少量字段：

```python
defaults=[
    "/experiment/pixeldit_text_to_image_finetune_res_2048",
    {"override /data_train": "pixeldit_MultiAspect_4K_1M_1bs_multires_2048_3840"},
    "_self_",
]
```

`_build_debug_run()` 也使用相同方式继承正式实验，然后将迭代次数、采样频率和 W&B 模式改为适合调试的值。对应实验名是在正式实验名后追加 `_debug`。

## 命令行 override

命令行可以同时选择配置组并覆盖最终字段。字段 override 的优先级高于实验中写入的值。

```bash
PYTHONPATH=. torchrun --nproc_per_node=4 --master_port=12341 -m scripts.train \
  --config=pid/_src/configs/pid_training/config.py \
  -- \
  experiment=pixeldit_text_to_image_finetune_res_2048 \
  trainer.max_iter=1000 \
  trainer.logging_iter=20 \
  optimizer.lr=2e-5 \
  checkpoint.load_path=/path/to/model.pth \
  job.wandb_mode=disabled
```

常见写法：

```bash
# 选择另一个已注册的数据配置；这个例子会将单卡 batch size 改成 2
data_train=pixeldit_MultiAspect_4K_1M_2bs_2048

# 覆盖布尔值
trainer.run_validation=true

# 覆盖列表；建议将包含 []、逗号等特殊字符的整项用单引号包住
'scheduler.warm_up_steps=[1000]'

# 同时选择多个 callback 配置
'callbacks=[basic,wandb]'
```

可以把最终优先级简化理解为：

```text
make_config() 基础值
  < 根 defaults 选择的基础组
  < 实验 defaults 中的组替换或父实验
  < 实验自身（实验中的 _self_ 位于最后）
  < 命令行字段 override
```

同一个 `defaults` 列表内部仍然按从前到后的顺序组合。

## 新增配置

### 新增组件选项

1. 用普通 `dict`、attrs 对象或 `LazyCall` 定义配置节点；
2. 使用 `ConfigStore.instance().store(group=..., name=..., package=..., node=...)` 注册；
3. 确保对应的 `register_*()` 在入口 `make_config()` 中被调用；
4. 在实验 `defaults` 或命令行中按组名选择该选项。

`LazyCall` 只保存 `_target_` 和构造参数，不会在导入配置时创建模型或 dataloader。

### 新增实验

1. 在已有实验包中定义一个 `LazyDict`，并合理安排 `defaults` 与 `_self_`；
2. 注册到 `group="experiment"`，通常使用 `package="_global_"`；
3. 确保模块会被入口导入。当前入口会递归导入已列出的实验包；如果新建了一个实验包，需要在 `make_config()` 中增加对应的 `import_all_modules_from_package()`。

实验是否可选取取决于模块是否被导入并执行了 `cs.store()`，而不只取决于 Python 变量名或文件名。

## 常见问题

- **提示 override 必须以 `--` 开头**：检查 `--config` 与 `experiment=...` 之间是否有独立的 `--`。
- **找不到某个组或选项**：检查 `cs.store()` 的 `group`/`name`，以及注册函数或实验模块是否被入口调用/导入。
- **覆盖路径没有生效**：确认使用的是最终 `package` 路径；例如网络字段位于 `model.config.net`，不是 `net`。
- **实验值被组件默认值覆盖**：检查 `_self_` 的位置。希望实验正文最后生效时，应将 `_self_` 放在实验 `defaults` 末尾。
- **新增顶层字段失败**：最终结果会被还原为结构化 attrs 配置，任意新增未声明的顶层字段可能被拒绝；优先覆盖已有字段或先扩展 `Config` 定义。
- **误用 Hydra multirun**：本项目没有暴露常规 `@hydra.main` 的 `-m/--multirun` 入口；命令中的 `-m scripts.train` 是 Python 模块启动参数。
