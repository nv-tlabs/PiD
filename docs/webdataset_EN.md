# WebDataset Multimodal Data Loading Mechanism

## Overview

This project's WebDataset implementation supports storing data from different modalities (such as images and captions) in **identically named tar files located in separate directories**. During loading, synchronized iteration aligns the data by `__key__`, and the aligned entries are then merged into complete data samples.

## Key Files

| File Path | Function |
|-----------|----------|
| `pid/_ext/imaginaire/datasets/webdataset/webdataset.py` | Top-level `Dataset` class responsible for parsing wdinfo, building the augmentor chain, and assembling the complete data-loading pipeline |
| `pid/_ext/imaginaire/datasets/webdataset/utils/iterators.py` | Core iterators, including functions such as `url_opener`, `tar_file_expander`, and `tarfile_samples`, which implement synchronized loading across multiple tar files |
| `pid/_ext/imaginaire/datasets/webdataset/config/schema.py` | Data structure definitions: `TarSample`, `Wdinfo`, `DatasetConfig`, and `DatasetInfo` |
| `pid/_ext/imaginaire/datasets/webdataset/utils/misc.py` | Utility functions: `remove_extensions_from_keys`, `skip_keys`, and `update_url` |
| `pid/_ext/imaginaire/datasets/webdataset/utils/stream.py` | S3 streaming reads; `RetryingStream` supports resuming after interruptions |
| `pid/_ext/imaginaire/datasets/webdataset/distributors/basic.py` | `ShardlistBasic` distributor, which assigns tar files by rank and worker |
| `pid/_ext/imaginaire/datasets/webdataset/distributors/parallel_sync_basic.py` | `ShardlistBasicParallelSync`, a distributor with support for context and tensor parallelism |
| `pid/_src/datasets/dataset_provider.py` | Application-layer dataset registration and `DatasetConfig` assembly |
| `pid/_ext/imaginaire/datasets/webdataset/decoders` | Directory containing decoders for various data types |


## Data Storage Structure

Data from different modalities is stored in separate directories. The tar filenames and the sample keys inside them remain consistent:

```
/data/
├── image/
│   └── part_000000/
│       ├── 000000.tar    # Contains: id_0.jpg, id_1.jpg, id_2.jpg, ...
│       ├── 000001.tar
│       └── ...
└── caption/
    └── part_000000/
        ├── 000000.tar    # Contains: id_0.json, id_1.json, id_2.json, ...
        └── ...
```

The .json file in the caption tar is a dict, saving {"prompt": str, "prompt_medium": str, "prompt_short": str}. prompt_medium and prompt_short are optional.


`TarSample.keys` specifies which directories to load (for example, `["image", "caption"]`). `TarSample.path` is the shared relative path of the tar file (for example, `part_000000/000000.tar`). The final path is constructed as `root/key/path`.

## Loading Process

### Step 1: `url_opener` — Open a Separate Tar Stream for Each Key

```python
# iterators.py: url_opener()
for data_key in url.keys:
    url_path_full = os.path.join(url.root, data_key, url.path)
    stream.append(gopen(url_path_full))
# Result: stream = [image_tar_stream, caption_tar_stream]
```

Each `TarSample` produces N streams, where N = `len(keys)`.

### Step 2: `tar_file_expander` — Expand Multiple Tar Files Synchronously and Align Them by Key

Two synchronization modes are available:

#### Simple Mode (`sample_keys_full_list = None`)

Multiple tar iterators are aligned directly using Python's `zip()`:

```python
tar_file_iterator_list = [iter(image_tar), iter(caption_tar)]
for sample in zip(*tar_file_iterator_list):
    # sample[0] = id_0.jpg (from image tar)
    # sample[1] = id_0.json (from caption tar)
    for key_idx, sample_key in enumerate(sample):
        yield process_sample(sample_key, url, key_idx)
```

**Requirement**: The order and number of files in every tar archive must match exactly.

#### Indexed Mode (`sample_keys_full_list` Points to a Parquet File)

An external index file is used for explicit alignment, allowing missing or out-of-order samples:

1. Load the Parquet index to obtain the list of valid keys: `["id_0", "id_1", "id_2", ...]`.
2. Maintain a `target_index` and call `run_iterator_to_index()` on each tar iterator to advance it to the target position.
3. Yield a sample only when **all tar files contain the corresponding sample**. Missing keys are skipped and recorded in the logs.

### Step 3: `process_sample` — Rewrite Filenames to Identify Their Sources

The data key is inserted into each original filename:

```
id_0.jpg  →  id_0.image.jpg
id_0.json  →  id_0.caption.json
```

This allows WebDataset's subsequent `group_by_keys` operation to group entries by their `__key__` prefix (`id_0`) and merge data from different modalities into a single dictionary:

```python
{"__key__": "id_0", "image.jpg": <bytes>, "caption.json": <bytes>}
```

### Step 4: Decode → Augmentation → Batch

In `webdataset.py`, `build_dataset()` adds the following stages in order:

1. **shuffle** — Shuffle sample order (with support for deterministic shuffling through `detshuffle`).
2. **decode** — Automatically decode data based on file suffixes (for example, jpg → PIL Image and pkl → Python object).
3. **remove_extensions_from_keys** — Remove suffixes from keys: `"image.jpg"` → `"image"`.
4. **skip_keys** — Filter out unneeded keys.
5. **augmentation** — Apply the data augmentation chain.
6. **update_url** — Update URL metadata.

## Distribution Mechanism

### ShardlistBasic

Tar file lists are sharded by `rank` and `worker_id`:

```
All tar files → split by node → split by worker → each worker receives a subset
```

This distributor supports shuffling and resuming from checkpoints through `set_epoch` and `set_resume_step`.

### ShardlistBasicParallelSync

This extends `ShardlistBasic` and ensures that ranks within the same context/tensor parallel group receive the same tar files by using `group_id` instead of `rank` for sharding.

## Key Data Structures

```python
@attrs.define
class TarSample:
    path: str                          # Relative tar path, e.g., "part_000000/000000.tar"
    root: str                          # Dataset root directory
    keys: list                         # Modality directories to load, e.g., ["image", "caption"]
    meta: DatasetInfo                  # Dataset metadata
    dset_id: str                       # Dataset ID
    sample_keys_full_list: str = None  # Optional Parquet index path (used for indexed synchronization)

@attrs.define
class Wdinfo:
    tar_files: list[TarSample]         # List of all tar files
    total_key_count: int               # Total number of samples
    chunk_size: int                    # Samples per tar file (used to size the shuffle buffer)
```
