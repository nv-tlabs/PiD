# Concise Guide to the PixelDiT Dataloader Configuration Group

## Overview

PixelDiT training selects a dataloader through Hydra's `data_train` configuration group. Three kinds of names need to be distinguished:

1. **data source**: a source name and its search directory for `wdinfo.json`;
2. **logical dataset**: a dataset composed of one or more sources;
3. **Hydra dataloader option**: a logical dataset combined with a batch size, resolution, and augmentor.

Their relationship is as follows:

```text
IMAGES_DATASET_SOURCES: source name -> wdinfo.json search directory
  -> IMAGES_DATASETS: dataset name -> one or more source names
  -> dataloader_pixeldit.py: bulk-register options across datasets, batch sizes, and resolutions
  -> Hydra: data_train=<option name>
  -> Final configuration: dataloader_train
```

## Key Files

| File | Purpose |
|---|---|
| `pid/_src/configs/pid_training/defaults/dataloader_pixeldit.py` | Generates and registers PixelDiT dataloader options in the configuration group |
| `pid/_src/datasets/data_sources/data_source_local.py` | Maps source names to local search directories for `wdinfo.json` |
| `pid/_src/datasets/data_sources/dataset_definition.py` | Defines logical datasets and the sources they contain |
| `pid/_src/datasets/data_sources/data_registration.py` | Resolves logical datasets into `DatasetInfo` objects and locates `wdinfo.json` files |
| `pid/_src/datasets/dataset_provider.py` | Creates the actual image WebDataset |
| `pid/_src/configs/pid_training/config.py` | Training configuration entry point that calls the `register_*()` functions |

## What Data Sources Are Currently Available?

Only one image source is currently registered:

```python
# data_source_local.py
IMAGES_DATASET_SOURCES = {
    "MultiAspect_4K_1M": "data/image_MultiAspect_4K_1M_webdataset/",
}
```

The path here is used to recursively search for `wdinfo.json`. When tar files are actually read, the loader uses the `root` recorded in each `wdinfo.json` file. Therefore, after moving the data, you must not only update the source search directory but also ensure that the `root` in each `wdinfo.json` remains valid.

There is also only one logical dataset, and it contains only the source above:

```python
# dataset_definition.py
IMAGES_DATASETS = {
    "MultiAspect_4K_1M": ["MultiAspect_4K_1M"],
}
```

The key in `IMAGES_DATASETS`, which is `MultiAspect_4K_1M` here, becomes the `dataset_name` component of the dataloader option name. Its value is a list of source names, so one logical dataset can combine multiple data sources.

There is also a `data_train=mock_image` option. It is registered by `pid/_src/configs/common/defaults/dataloader.py` and generates random mock data; it does not correspond to a WebDataset source. The two PixelDiT registration functions currently create only `data_train` options; `data_val` currently has only `mock_image`.

`Rendered_Text`, `Nano_Banana_Image`, and `MultiAspect_4K_1M_plus` in `data_source_local.py` and `dataset_definition.py` are only commented-out examples and cannot currently be selected directly.

## How the `data_train` Group Is Created

The training entry point's `make_config()` calls:

```python
register_training_and_val_data()                 # Register mock_image
register_text_to_image_data()                    # Register fixed-resolution configurations
register_text_to_image_multi_resolution_data()   # Register multi-resolution configurations
```

Both PixelDiT functions register configurations in the same general way:

```python
cs.store(
    group="data_train",
    package="dataloader_train",
    name=option_name,
    node=loader_config,
)
```

Therefore:

- the Hydra group name is `data_train`;
- users select a configuration with `data_train=<option_name>`;
- the selected configuration is merged into `dataloader_train` in the final configuration;
- the registered node is a `LazyCall`, so registration does not access data or instantiate a dataloader.

## How Option Names Are Generated

### Fixed Resolution

`register_text_to_image_data()` iterates over the Cartesian product of the following three collections:

```python
dataset_name = list(IMAGES_DATASETS.keys())
batch_size = [1, 2, 4, 8, 12, 16, 32, 64]
resolution = ["1024", "2048", "3072", "3840", "4096"]
```

The naming pattern is:

```text
pixeldit_{dataset_name}_{batch_size}bs_{resolution}
```

For example:

```text
pixeldit_MultiAspect_4K_1M_1bs_2048
pixeldit_MultiAspect_4K_1M_4bs_1024
```

The numeric value in `Nbs` is assigned directly to each training process's `dataloader_train.batch_size`; it is not the global batch size for the entire multi-GPU job.

These configurations use `image_caption_augmentor`. Because there is currently only one logical dataset, `1 × 8 × 5 = 40` fixed-resolution options are registered.

### Multi-Resolution

`register_text_to_image_multi_resolution_data()` uses the same datasets and batch sizes, but supports only the following upper bounds:

```python
upper_bound = ["3072", "3840", "4096"]
```

The naming pattern is:

```text
pixeldit_{dataset_name}_{batch_size}bs_multires_2048_{upper_bound}
```

For example:

```text
pixeldit_MultiAspect_4K_1M_1bs_multires_2048_3840
```

These configurations use `image_caption_multi_resolution_augmentor`. `multires_2048_3840` does not choose only between 2048 and 3840. Based on the original image dimensions, it selects the largest grid level between 2048 and 3840 that the sample can accommodate.

Currently, `1 × 8 × 3 = 24` multi-resolution options are registered. Including `mock_image`, the `data_train` group currently has 65 registered options in total.

## How to List the Exact Option Names

These options are registered dynamically at Python runtime, so there are no YAML files that can be listed directly with `ls`. Run the following command from the repository root:

```bash
PYTHONPATH=. python - <<'PY'
from hydra.core.config_store import ConfigStore
from pid._src.configs.pid_training.config import make_config

make_config()
for option in sorted(ConfigStore.instance().list("data_train")):
    print(option.removesuffix(".yaml"))
PY
```

You must call `make_config()` first so that every `register_*()` function is executed. The names returned by `ConfigStore.list()` end in `.yaml`, but these are only Hydra's internal virtual configuration names; omit `.yaml` when selecting an option.

To list only the PixelDiT options, add a filter:

```python
for option in sorted(ConfigStore.instance().list("data_train")):
    option = option.removesuffix(".yaml")
    if option.startswith("pixeldit_"):
        print(option)
```

This command confirms only that the configuration is registered. It does not prove that the source search directory, `wdinfo.json` files, and tar data are all readable, because the data is accessed only when the dataloader is instantiated for training.

## How to Select a Dataloader

Select an option in the experiment configuration's `defaults`:

```python
defaults=[
    {"override /data_train": "pixeldit_MultiAspect_4K_1M_1bs_2048"},
    "_self_",
]
```

The root configuration has already selected `data_train: mock_image`, so the experiment uses `override /data_train` to replace it.

You can also override the experiment's selection from the command line:

```bash
PYTHONPATH=. torchrun --nproc_per_node=4 --master_port=12341 -m scripts.train \
  --config=pid/_src/configs/pid_training/config.py \
  -- \
  experiment=pixeldit_text_to_image_finetune_res_2048 \
  data_train=pixeldit_MultiAspect_4K_1M_2bs_2048 \
  dataloader_train.num_workers=8
```

The two overrides have different meanings:

- `data_train=...` selects a Hydra configuration-group option;
- `dataloader_train.num_workers=8` overrides a specific field in the final configuration after group selection.

After selecting `pixeldit_MultiAspect_4K_1M_2bs_2048`, the key configuration fields are equivalent to:

```yaml
dataloader_train:
  batch_size: 2
  dataset:
    dataset_name: MultiAspect_4K_1M
    resolution: "2048"
    augmentor_name: image_caption_augmentor
    input_keys: [image, caption]
```

## Adding a New Data Source

First, convert the data to the WebDataset format required by this project and ensure that `wdinfo.json` can be found recursively under the directory. The PixelDiT loader reads the `image` and `caption` keys by default; see [webdataset_EN.md](webdataset_EN.md) for the detailed format.

Then register the source and its search directory for `wdinfo.json` in `data_source_local.py`:

```python
IMAGES_DATASET_SOURCES = {
    "MultiAspect_4K_1M": "data/image_MultiAspect_4K_1M_webdataset/",
    "Rendered_Text": "data/image_Rendered_Text_webdataset/",
}
```

Next, define a logical dataset in `dataset_definition.py`. It can contain a single source or combine multiple sources:

```python
IMAGES_DATASETS = {
    "MultiAspect_4K_1M": ["MultiAspect_4K_1M"],
    "MultiAspect_4K_1M_plus": [
        "MultiAspect_4K_1M",
        "Rendered_Text",
    ],
}
```

After restarting the Python process, the registration loops automatically generate options such as:

```text
pixeldit_MultiAspect_4K_1M_plus_1bs_2048
pixeldit_MultiAspect_4K_1M_plus_1bs_multires_2048_3840
```

You do not need to write a separate `cs.store()` call for every batch size and resolution. Source names are case-sensitive, and referencing an unregistered source from a logical dataset raises a `KeyError` at runtime.

## Notes on the Current Implementation

- `ConfigStore.list()` and `--dryrun` check only configuration composition; they do not validate the data itself. A nonexistent source search path, the absence of a matching `wdinfo.json` file, or incomplete data keys will still cause a failure when the dataloader is instantiated or the data is read.
- Every batch size is registered, but large batch sizes at high resolutions may cause OOM errors. The existence of an option does not guarantee that the corresponding configuration will run on the available hardware.
- Relative search paths in `data_source_local.py` are resolved against the current working directory when training starts, while the actual tar root comes from the `root` in `wdinfo.json`. We recommend starting training from the repository root; when moving data, inspect or rebuild the `wdinfo.json` files, and restart the Python process after modifying source or dataset definitions.

For details about Hydra `defaults`, configuration groups, and override rules, see [hydra_EN.md](hydra_EN.md).
