# PixelDiT Dataloader 配置组简明教程

## 概述

PixelDiT 训练使用 Hydra 的 `data_train` 配置组选择 dataloader。这里需要区分三种名称：

1. **data source**：一个 source name 及其 `wdinfo.json` 搜索目录；
2. **逻辑 dataset**：由一个或多个 source 组成；
3. **Hydra dataloader 选项**：在逻辑 dataset 的基础上，再组合 batch size、分辨率和 augmentor。

它们之间的关系是：

```text
IMAGES_DATASET_SOURCES：source name -> wdinfo.json 搜索目录
  -> IMAGES_DATASETS：dataset name -> 一个或多个 source name
  -> dataloader_pixeldit.py：按 dataset、batch size、分辨率批量注册
  -> Hydra：data_train=<选项名>
  -> 最终配置：dataloader_train
```

## 关键文件

| 文件 | 作用 |
|---|---|
| `pid/_src/configs/pid_training/defaults/dataloader_pixeldit.py` | 生成并注册 PixelDiT dataloader 配置组选项 |
| `pid/_src/datasets/data_sources/data_source_local.py` | source name 到本地 `wdinfo.json` 搜索目录的映射 |
| `pid/_src/datasets/data_sources/dataset_definition.py` | 定义逻辑 dataset 及其包含的 source |
| `pid/_src/datasets/data_sources/data_registration.py` | 将逻辑 dataset 解析为 `DatasetInfo`，并查找 `wdinfo.json` |
| `pid/_src/datasets/dataset_provider.py` | 创建实际的 image WebDataset |
| `pid/_src/configs/pid_training/config.py` | 调用各个 `register_*()` 的训练配置入口 |

## 当前有哪些数据源

当前真正登记的图像 source 只有一个：

```python
# data_source_local.py
IMAGES_DATASET_SOURCES = {
    "MultiAspect_4K_1M": "data/image_MultiAspect_4K_1M_webdataset/",
}
```

这里的路径用于递归查找 `wdinfo.json`；实际读取 tar 时使用的是每个 `wdinfo.json` 内部记录的 `root`。因此移动数据后，除了修改 source 搜索目录，还要保证 `wdinfo.json` 中的 `root` 仍然有效。

当前逻辑 dataset 也只有一个，并且只包含上面的 source：

```python
# dataset_definition.py
IMAGES_DATASETS = {
    "MultiAspect_4K_1M": ["MultiAspect_4K_1M"],
}
```

`IMAGES_DATASETS` 的 key，也就是这里的 `MultiAspect_4K_1M`，会成为 dataloader 选项名中的 `dataset_name`。它的 value 是 source name 列表，因此一个逻辑 dataset 可以组合多个数据源。

另外还有一个 `data_train=mock_image`，它由 `pid/_src/configs/common/defaults/dataloader.py` 注册，用于生成随机 mock 数据，不属于 WebDataset source。当前两个 PixelDiT 注册函数只创建 `data_train` 选项；`data_val` 目前只有 `mock_image`。

`data_source_local.py` 和 `dataset_definition.py` 中的 `Rendered_Text`、`Nano_Banana_Image`、`MultiAspect_4K_1M_plus` 都只是注释示例，当前不能直接选择。

## `data_train` 组如何创建

训练入口的 `make_config()` 会调用：

```python
register_training_and_val_data()                 # 注册 mock_image
register_text_to_image_data()                    # 注册固定分辨率配置
register_text_to_image_multi_resolution_data()   # 注册多分辨率配置
```

两个 PixelDiT 函数都会执行类似的注册：

```python
cs.store(
    group="data_train",
    package="dataloader_train",
    name=option_name,
    node=loader_config,
)
```

因此：

- Hydra 组名是 `data_train`；
- 用户使用 `data_train=<option_name>` 选择配置；
- 选中的内容最终合入 `dataloader_train`；
- 注册的 node 是 `LazyCall`，注册时不会读取数据或创建 dataloader。

## 选项名称如何生成

### 固定分辨率

`register_text_to_image_data()` 遍历下面三个集合的笛卡尔积：

```python
dataset_name = list(IMAGES_DATASETS.keys())
batch_size = [1, 2, 4, 8, 12, 16, 32, 64]
resolution = ["1024", "2048", "3072", "3840", "4096"]
```

名称公式是：

```text
pixeldit_{dataset_name}_{batch_size}bs_{resolution}
```

例如：

```text
pixeldit_MultiAspect_4K_1M_1bs_2048
pixeldit_MultiAspect_4K_1M_4bs_1024
```

名称中的 `Nbs` 会直接写入每个训练进程的 `dataloader_train.batch_size`，不是整个多卡任务的全局 batch size。

这类配置使用 `image_caption_augmentor`。当前只有一个逻辑 dataset，所以共注册 `1 × 8 × 5 = 40` 个固定分辨率选项。

### 多分辨率

`register_text_to_image_multi_resolution_data()` 使用相同的 dataset 和 batch size，但上限只有：

```python
upper_bound = ["3072", "3840", "4096"]
```

名称公式是：

```text
pixeldit_{dataset_name}_{batch_size}bs_multires_2048_{upper_bound}
```

例如：

```text
pixeldit_MultiAspect_4K_1M_1bs_multires_2048_3840
```

这类配置使用 `image_caption_multi_resolution_augmentor`。`multires_2048_3840` 不是只在 2048 和 3840 两个尺寸之间二选一，而是根据原图尺寸，在 2048 到 3840 的多个网格级别中选择该样本能够容纳的最大级别。

当前共注册 `1 × 8 × 3 = 24` 个多分辨率选项。加上 `mock_image`，`data_train` 组目前共有 65 个已注册选项。

## 如何查看所有准确的选项名

这些选项是 Python 运行时动态注册的，不存在可直接 `ls` 的 YAML 文件。可以从仓库根目录运行：

```bash
PYTHONPATH=. python - <<'PY'
from hydra.core.config_store import ConfigStore
from pid._src.configs.pid_training.config import make_config

make_config()
for option in sorted(ConfigStore.instance().list("data_train")):
    print(option.removesuffix(".yaml"))
PY
```

必须先调用 `make_config()`，这样所有 `register_*()` 才会执行。`ConfigStore.list()` 返回的名称带 `.yaml` 后缀，但这只是 Hydra 的内部虚拟名称；实际选择时不要带 `.yaml`。

只查看 PixelDiT 选项时，可以增加过滤：

```python
for option in sorted(ConfigStore.instance().list("data_train")):
    option = option.removesuffix(".yaml")
    if option.startswith("pixeldit_"):
        print(option)
```

这条命令只能确认配置已经注册，不能证明 source 搜索目录、`wdinfo.json` 和 tar 数据都能正常读取，因为数据是在训练实例化 dataloader 时才真正访问的。

## 如何选择 dataloader

在实验配置的 `defaults` 中选择：

```python
defaults=[
    {"override /data_train": "pixeldit_MultiAspect_4K_1M_1bs_2048"},
    "_self_",
]
```

根配置已经选择了 `data_train: mock_image`，所以实验中使用 `override /data_train` 将它替换掉。

也可以在命令行覆盖实验自己的选择：

```bash
PYTHONPATH=. torchrun --nproc_per_node=4 --master_port=12341 -m scripts.train \
  --config=pid/_src/configs/pid_training/config.py \
  -- \
  experiment=pixeldit_text_to_image_finetune_res_2048 \
  data_train=pixeldit_MultiAspect_4K_1M_2bs_2048 \
  dataloader_train.num_workers=8
```

这里的两种 override 含义不同：

- `data_train=...`：选择 Hydra 配置组选项；
- `dataloader_train.num_workers=8`：选组完成后，覆盖最终配置中的具体字段。

选择 `pixeldit_MultiAspect_4K_1M_2bs_2048` 后，关键配置相当于：

```yaml
dataloader_train:
  batch_size: 2
  dataset:
    dataset_name: MultiAspect_4K_1M
    resolution: "2048"
    augmentor_name: image_caption_augmentor
    input_keys: [image, caption]
```

## 添加新的 data source

首先将数据转换为项目所需的 WebDataset 格式，并确保目录下能够递归找到 `wdinfo.json`。PixelDiT loader 默认读取 `image` 和 `caption` 两种 key；具体格式见 [webdataset_CN.md](webdataset_CN.md)。

然后在 `data_source_local.py` 中注册 source 及其 `wdinfo.json` 搜索目录：

```python
IMAGES_DATASET_SOURCES = {
    "MultiAspect_4K_1M": "data/image_MultiAspect_4K_1M_webdataset/",
    "Rendered_Text": "data/image_Rendered_Text_webdataset/",
}
```

再在 `dataset_definition.py` 中定义逻辑 dataset。它可以只包含一个 source，也可以组合多个 source：

```python
IMAGES_DATASETS = {
    "MultiAspect_4K_1M": ["MultiAspect_4K_1M"],
    "MultiAspect_4K_1M_plus": [
        "MultiAspect_4K_1M",
        "Rendered_Text",
    ],
}
```

重新启动 Python 进程后，注册循环会自动生成例如：

```text
pixeldit_MultiAspect_4K_1M_plus_1bs_2048
pixeldit_MultiAspect_4K_1M_plus_1bs_multires_2048_3840
```

不需要为每个 batch size 和分辨率手写 `cs.store()`。注意 source name 大小写必须完全一致，逻辑 dataset 引用未注册的 source 会在运行时触发 `KeyError`。

## 当前实现的注意事项

- `ConfigStore.list()` 和 `--dryrun` 只能检查配置组合，不能验证数据内容。source 搜索路径不存在、找不到匹配的 `wdinfo.json` 或数据 key 不完整，仍会在实例化或读取数据时失败。
- 所有 batch size 都会被注册，但高分辨率下的大 batch size 可能 OOM；选项存在不代表硬件一定能够运行。
- `data_source_local.py` 中的相对搜索路径按启动训练时的当前目录解析，实际 tar 根目录则来自 `wdinfo.json` 的 `root`。建议从仓库根目录启动；移动数据时检查或重建 wdinfo，并在修改 source 或 dataset 定义后重新启动 Python 进程。

Hydra 的 `defaults`、配置组和 override 规则详见 [hydra_CN.md](hydra_CN.md)。
