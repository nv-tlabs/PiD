# WebDataset 多模态数据加载机制

## 概述

本项目的 WebDataset 实现支持将不同模态的数据（如 image、caption）存储在**不同目录下的同名 tar 文件**中，加载时通过同步迭代按 `__key__` 对齐，最终合并为完整的 data sample。

## 关键文件

| 文件路径 | 功能 |
|---------|------|
| `pid/_ext/imaginaire/datasets/webdataset/webdataset.py` | 顶层 `Dataset` 类，负责解析 wdinfo、构建 augmentor 链、组装完整的数据加载 pipeline |
| `pid/_ext/imaginaire/datasets/webdataset/utils/iterators.py` | 核心迭代器，包含 `url_opener`、`tar_file_expander`、`tarfile_samples` 等函数，实现多 tar 同步加载 |
| `pid/_ext/imaginaire/datasets/webdataset/config/schema.py` | 数据结构定义：`TarSample`、`Wdinfo`、`DatasetConfig`、`DatasetInfo` |
| `pid/_ext/imaginaire/datasets/webdataset/utils/misc.py` | 工具函数：`remove_extensions_from_keys`、`skip_keys`、`update_url` |
| `pid/_ext/imaginaire/datasets/webdataset/utils/stream.py` | S3 流式读取，`RetryingStream` 支持断点重试 |
| `pid/_ext/imaginaire/datasets/webdataset/distributors/basic.py` | `ShardlistBasic` 分发器，按 rank/worker 分配 tar 文件 |
| `pid/_ext/imaginaire/datasets/webdataset/distributors/parallel_sync_basic.py` | `ShardlistBasicParallelSync`，支持 context/tensor parallelism 的分发器 |
| `pid/_src/datasets/dataset_provider.py` | 业务层数据注册和 DatasetConfig 组装 |
| `pid/_ext/imaginaire/datasets/webdataset/decoders` | decoders 目录，包含各种数据解码器 |


## 数据存储结构

不同模态的数据存放在不同目录下，tar 文件名和内部 sample key 保持一致：

```
/data/
├── image/
│   └── part_000000/
│       ├── 000000.tar    # 包含: id_0.jpg, id_1.jpg, id_2.jpg, ...
│       ├── 000001.tar
│       └── ...
└── caption/
    └── part_000000/
        ├── 000000.tar    # 包含: id_0.json, id_1.json, id_2.json, ...
        └── ...
```

caption的.json 文件为dict, 保存{"prompt": str, "prompt_medium": str, "prompt_short": str}. prompt_medium 和 prompt_short 是可选的。

`TarSample.keys` 定义了要加载哪些目录（如 `["image", "caption"]`），`TarSample.path` 是共享的 tar 相对路径（如 `part_000000/000000.tar`），最终拼接为 `root/key/path`。

## 加载流程

### 第一步：`url_opener` — 为每个 key 打开独立的 tar stream

```python
# iterators.py: url_opener()
for data_key in url.keys:
    url_path_full = os.path.join(url.root, data_key, url.path)
    stream.append(gopen(url_path_full))
# 结果: stream = [image_tar_stream, caption_tar_stream]
```

每个 `TarSample` 会产生 N 个 stream（N = len(keys)）。

### 第二步：`tar_file_expander` — 同步展开多个 tar 并按 key 对齐

提供两种同步模式：

#### 简单模式（`sample_keys_full_list = None`）

直接用 Python `zip()` 对齐多个 tar 迭代器：

```python
tar_file_iterator_list = [iter(image_tar), iter(caption_tar)]
for sample in zip(*tar_file_iterator_list):
    # sample[0] = id_0.jpg (from image tar)
    # sample[1] = id_0.json (from caption tar)
    for key_idx, sample_key in enumerate(sample):
        yield process_sample(sample_key, url, key_idx)
```

**前提**：所有 tar 内文件顺序和数量必须完全一致。

#### 索引模式（`sample_keys_full_list` 指向 parquet 文件）

通过外部索引文件显式对齐，可容忍缺失和乱序：

1. 加载 parquet 索引得到合法 key 列表：`["id_0", "id_1", "id_2", ...]`
2. 维护 `target_index`，对每个 tar 迭代器调用 `run_iterator_to_index()` 读取到目标位置
3. 只有当**所有 tar 都找到了对应 sample** 时才 yield，缺失的 key 会被跳过并记录日志

### 第三步：`process_sample` — 重写文件名以标记来源

将原始文件名注入 data_key 信息：

```
id_0.jpg  →  id_0.image.jpg
id_0.json  →  id_0.caption.json
```

这样后续 webdataset 的 `group_by_keys` 按 `__key__` 前缀（`id_0`）分组时，可以将不同模态合并到同一个 dict 中：

```python
{"__key__": "id_0", "image.jpg": <bytes>, "caption.json": <bytes>}
```

### 第四步：decode → augmentation → batch

在 `webdataset.py` 的 `build_dataset()` 中依次添加：

1. **shuffle** — 打乱 sample 顺序（支持 `detshuffle` 确定性打乱）
2. **decode** — 根据后缀自动解码（如 jpg → PIL Image, pkl → Python object）
3. **remove_extensions_from_keys** — 去掉后缀，`"image.jpg"` → `"image"`
4. **skip_keys** — 过滤不需要的 key
5. **augmentation** — 数据增强链
6. **update_url** — 更新 URL 元信息

## 分发机制

### ShardlistBasic

按 `rank` 和 `worker_id` 将 tar 文件列表分片：

```
全部 tar 文件 → 按 node 分 → 按 worker 分 → 每个 worker 得到一个子集
```

支持 shuffle 和断点恢复（通过 `set_epoch` 和 `set_resume_step`）。

### ShardlistBasicParallelSync

扩展 ShardlistBasic，确保同一个 context/tensor parallel group 内的 rank 拿到相同的 tar 文件（通过 `group_id` 替代 `rank` 进行分片）。

## 关键数据结构

```python
@attrs.define
class TarSample:
    path: str                          # tar 文件相对路径，如 "part_000000/000000.tar"
    root: str                          # 数据根目录
    keys: list                         # 要加载的模态目录列表，如 ["image", "caption"]
    meta: DatasetInfo                  # 数据集元信息
    dset_id: str                       # 数据集 ID
    sample_keys_full_list: str = None  # 可选的 parquet 索引文件路径（用于索引模式同步）

@attrs.define
class Wdinfo:
    tar_files: list[TarSample]         # 所有 tar 文件列表
    total_key_count: int               # 总 sample 数
    chunk_size: int                    # 每个 tar 内的 sample 数（用于 shuffle buffer 大小）
```
