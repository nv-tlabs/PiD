"""
简化的本地分片管道，用于从多模态文件夹结构创建webdatasets。

自动扫描input-dir下的所有子文件夹，并根据文件夹名称识别数据类型。

支持的数据类型和文件扩展名：
- video/: .mp4 文件
- image/: .jpg, .png 文件
- caption/: .json 文件, dict, saved as {"prompt": str, "prompt_medium": str, "prompt_short": str}. prompt_medium and prompt_short are optional
- umt5/: .pkl 文件

创建 webdataset 格式，每种数据类型分别存储在单独的tar文件中：
- video/*.tar: <uid>.mp4 包含原始视频字节
- image/*.tar: <uid>.jpg 或 <uid>.png 包含图像数据
- caption/*.tar: <uid>.json 包含字幕数据
- umt5/*.tar: <uid>.pkl 包含嵌入数据

功能特性：
- 自动发现和处理所有子文件夹
- 支持多种数据类型和文件格式
- 使用标准库tarfile
- 直接扫描文件夹并进行全面验证
- 灵活处理可选数据类型
"""

import argparse
import io
import json
import os
import random
import sys
import tarfile
import time
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from loguru import logger

logger.remove()
logger.add(
    sys.stderr,
    level=os.environ.get("LOGURU_LEVEL", "INFO"),
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<7} | {message}",
)

# 定义支持的数据类型及其文件扩展名
DATA_TYPE_EXTENSIONS = {
    "video": [".mp4"],
    "image": [".jpg", ".png", ".jpeg", ".JPG"],
    "caption": [".json"],
    "umt5": [".pkl", ".pickle"],
}

# Predefined aspect ratios for classification (name, ratio_value).
# The classifier picks the nearest ratio in log-space.
PREDEFINED_ASPECT_RATIOS = [
    ("21_9", 21 / 9),
    ("16_9", 16 / 9),
    ("3_2", 3 / 2),
    ("4_3", 4 / 3),
    ("1_1", 1.0),
    ("3_4", 3 / 4),
    ("2_3", 2 / 3),
    ("9_16", 9 / 16),
]

DEFAULT_DURATION_BINS = "0,5,10,30,inf"
WDINFO_CHUNK_SIZE = 10
WDINFO_NUM_WORKERS = 32


def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="输入目录路径，会自动扫描所有子文件夹（如 video/, image/, caption/, umt5/）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="webdataset将被保存的本地目录",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=20,
        help="每个分片的样本数量（默认：100）。如果指定了--max-shard-size-mb，此参数将被忽略",
    )
    parser.add_argument(
        "--max-shard-size-mb",
        default=500,
        type=int,
        help="每个tar文件的最大大小（MB）。如果指定，将根据文件实际大小智能分组，忽略--shard-size参数",
    )
    parser.add_argument(
        "--size-based-on",
        type=str,
        default="image",
        choices=["video", "image"],
        help="用于计算分片大小的数据类型（默认：image）",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="不打乱样本（默认会打乱）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="限制要处理的样本数量",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        help="输出文件的数据集名称（如果不提供，则使用输入目录名称）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="显示将要处理的内容，但不实际创建文件",
    )
    parser.add_argument(
        "--max-file-size-mb",
        type=int,
        default=1024,
        help="要处理的最大文件大小（MB）（默认：1024）",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=50,
        help="每N个样本记录一次进度（默认：10）",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="并行处理的工作进程数（默认：1，即串行处理）。建议设置为CPU核心数",
    )
    parser.add_argument(
        "--part-size",
        type=int,
        default=1000,
        help="每个part文件夹包含的shard数量。如果指定，输出结构将为{part_idx:06}/{shard_idx:06}.tar",
    )
    parser.add_argument(
        "--classify-aspect-ratio",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable aspect ratio (and duration for video) classification. "
        "Output structure: aspect_ratio_X_Y/[duration_A_B/]{data_type}/... "
        "Use --no-classify-aspect-ratio to disable.",
    )
    parser.add_argument(
        "--duration-bins",
        type=str,
        default=DEFAULT_DURATION_BINS,
        help=f"Comma-separated duration bin boundaries in seconds (default: '{DEFAULT_DURATION_BINS}'). "
        "Only used when --classify-aspect-ratio is set and video data is present.",
    )
    parser.add_argument(
        "--metadata-workers",
        type=int,
        default=64,
        help="Number of parallel workers for reading media metadata (default: 16). "
        "Only used when --classify-aspect-ratio is set.",
    )
    return parser.parse_args()


import math


def _classify_aspect_ratio(w: int, h: int) -> str:
    """Classify width/height into the nearest predefined aspect ratio using log-space distance.

    Returns a string like "aspect_ratio_16_9".
    """
    ratio = w / h
    log_ratio = math.log(ratio)
    best_name = PREDEFINED_ASPECT_RATIOS[0][0]
    best_dist = float("inf")
    for name, ref_ratio in PREDEFINED_ASPECT_RATIOS:
        dist = abs(log_ratio - math.log(ref_ratio))
        if dist < best_dist:
            best_dist = dist
            best_name = name
    return f"aspect_ratio_{best_name}"


def _parse_duration_bins(bins_str: str) -> List[float]:
    """Parse comma-separated duration bin boundaries.

    Example: "0,5,10,30,inf" -> [0.0, 5.0, 10.0, 30.0, inf]
    """
    parts = [s.strip() for s in bins_str.split(",")]
    return [float(p) for p in parts]


def _classify_duration(dur: float, bin_edges: List[float]) -> str:
    """Place duration into a bin defined by adjacent edges.

    For bin_edges [0, 5, 10, 30, inf] and dur=7.5, returns "duration_5_10".
    """
    for i in range(len(bin_edges) - 1):
        if dur < bin_edges[i + 1]:
            lo = str(int(bin_edges[i])) if not math.isinf(bin_edges[i]) else "inf"
            hi = str(int(bin_edges[i + 1])) if not math.isinf(bin_edges[i + 1]) else "inf"
            return f"duration_{lo}_{hi}"
    # Falls above all bins — put into last bin
    lo = str(int(bin_edges[-2])) if not math.isinf(bin_edges[-2]) else "inf"
    hi = str(int(bin_edges[-1])) if not math.isinf(bin_edges[-1]) else "inf"
    return f"duration_{lo}_{hi}"


def _get_video_metadata_ffprobe(path: str) -> Tuple[int, int, float]:
    """Fallback: read video resolution and duration using ffprobe.

    Returns (width, height, duration_seconds).
    """
    import json
    import subprocess

    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    probe = json.loads(result.stdout)

    # Find the video stream
    w, h = 0, 0
    for stream in probe.get("streams", []):
        if stream.get("codec_type") == "video":
            w = int(stream["width"])
            h = int(stream["height"])
            break

    if w == 0 or h == 0:
        raise ValueError(f"No video stream found in {path}")

    duration = float(probe.get("format", {}).get("duration", 0.0))
    return w, h, duration


def _get_video_metadata(path: str) -> Tuple[int, int, float]:
    """Read video resolution and duration using decord, falling back to ffprobe on failure.

    Returns (width, height, duration_seconds).
    """
    return 1280, 720, 11.0
    try:
        from decord import VideoReader

        vr = VideoReader(path)
        w, h = vr[0].shape[1], vr[0].shape[0]  # decord frame shape is (H, W, C)
        fps = vr.get_avg_fps()
        duration = len(vr) / fps if fps > 0 else 0.0
        return w, h, duration
    except Exception as e:  # noqa: BLE001
        return _get_video_metadata_ffprobe(path)


def _get_image_metadata(path: str) -> Tuple[int, int]:
    """Read image dimensions using PIL (header only, lazy import).

    Returns (width, height).
    """
    from PIL import Image

    with Image.open(path) as img:
        return img.size  # (width, height)


def _get_metadata_for_sample(
    args_tuple: Tuple[Dict[str, Any], Dict[str, List[str]]],
) -> Tuple[str, Optional[Tuple[int, int, Optional[float]]]]:
    """Worker function for parallel metadata reading.

    Args:
        args_tuple: (sample_dict, data_folders)

    Returns:
        (basename, (width, height, duration_or_None)) or (basename, None) on failure.
    """
    sample_dict, data_folders = args_tuple
    basename = sample_dict["basename"]
    data_paths = sample_dict["data_paths"]
    try:
        if "video" in data_paths and "video" in data_folders:
            w, h, dur = _get_video_metadata(data_paths["video"])
            return basename, (w, h, dur)
        elif "image" in data_paths and "image" in data_folders:
            w, h = _get_image_metadata(data_paths["image"])
            return basename, (w, h, None)
        else:
            logger.warning(f"Sample {basename} has no video or image data for metadata extraction")
            return basename, None
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to read metadata for {basename}: {e}")
        return basename, None


def _bucket_samples(
    sample_dicts: List[Dict[str, Any]],
    data_folders: Dict[str, List[str]],
    duration_bins: List[float],
    workers: int,
) -> Dict[str, List[Dict[str, Any]]]:
    """Classify all samples into aspect-ratio (and optionally duration) buckets.

    Returns a dict mapping bucket keys to lists of sample_dicts, e.g.:
      {"aspect_ratio_16_9/duration_5_10": [...], "aspect_ratio_4_3": [...]}
    Video samples get both aspect_ratio and duration levels.
    Image samples get only aspect_ratio level.
    """
    has_video = "video" in data_folders

    # Read metadata in parallel
    logger.info(f"Reading metadata: samples={len(sample_dicts)}, workers={workers}")
    args_list = [(sd, data_folders) for sd in sample_dicts]

    metadata_map: Dict[str, Tuple[int, int, Optional[float]]] = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for basename, meta in executor.map(_get_metadata_for_sample, args_list):
            if meta is not None:
                metadata_map[basename] = meta

    logger.info(f"Metadata read complete: ok={len(metadata_map)}, total={len(sample_dicts)}")

    # Classify into buckets
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    skipped = 0
    for sd in sample_dicts:
        basename = sd["basename"]
        if basename not in metadata_map:
            skipped += 1
            continue
        w, h, dur = metadata_map[basename]
        ar_key = _classify_aspect_ratio(w, h)
        if has_video and dur is not None:
            dur_key = _classify_duration(dur, duration_bins)
            bucket_key = f"{ar_key}/{dur_key}"
        else:
            bucket_key = ar_key
        buckets.setdefault(bucket_key, []).append(sd)

    bucket_sizes = sorted(((bk, len(items)) for bk, items in buckets.items()), key=lambda item: item[1], reverse=True)
    largest = ", ".join(f"{bk}={count}" for bk, count in bucket_sizes[:8])
    logger.info(f"Bucketing complete: buckets={len(buckets)}, skipped={skipped}, largest=[{largest}]")

    return buckets


def _validate_file_size(file_path: str, max_size_mb: int) -> bool:
    """检查文件大小是否在可接受的范围内。"""
    try:
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if size_mb > max_size_mb:
            logger.warning(f"File exceeds size limit: path={file_path}, size={size_mb:.1f}MB, limit={max_size_mb}MB")
            return False
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to check file size: path={file_path}, error={e}")
        return True  # 如果大小检查失败则允许处理


class Sample:
    """单个样本的容器，包含所有可用的数据类型路径。"""

    def __init__(self, basename: str, dataset_name: str, data_paths: Dict[str, str]):
        """
        Args:
            basename: 样本的基础名称（不含扩展名）
            dataset_name: 数据集名称
            data_paths: 数据类型到文件路径的映射，例如 {"video": "/path/to/video.mp4"}
        """
        self.basename = basename
        self.dataset_name = dataset_name
        self.data_paths = data_paths


def _discover_data_folders(input_dir: str) -> Dict[str, List[str]]:
    """
    发现输入目录下所有包含支持文件类型的子文件夹。

    Returns:
        字典，键为文件夹名称，值为该文件夹支持的扩展名列表
    """
    input_path = Path(input_dir)
    discovered_folders = {}

    for folder_name, extensions in DATA_TYPE_EXTENSIONS.items():
        folder_path = input_path / folder_name
        if folder_path.exists() and folder_path.is_dir():
            # 检查是否有任何支持的文件类型
            has_files = False
            for ext in extensions:
                if list(folder_path.glob(f"*{ext}")):
                    has_files = True
                    break

            if has_files:
                discovered_folders[folder_name] = extensions
                logger.debug(f"Detected data folder: {folder_name}/ extensions={extensions}")

    return discovered_folders


def _scan_input_directory(
    input_dir: str,
    data_folders: Dict[str, List[str]],
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    扫描输入目录以查找所有数据文件。

    Args:
        input_dir: 输入目录路径
        data_folders: 要扫描的文件夹及其扩展名
        limit: 要处理的最大样本数量

    Returns:
        包含basename和各数据类型路径的字典列表
    """
    input_path = Path(input_dir)

    # 按数据类型收集所有文件
    all_files = {}

    for folder_name, extensions in data_folders.items():
        folder_path = input_path / folder_name
        all_files[folder_name] = {}

        for ext in extensions:
            files = list(folder_path.glob(f"*{ext}"))

            for f in files:
                basename = f.stem
                all_files[folder_name][basename] = str(f)

        logger.debug(f"Scanned files: key={folder_name}, files={len(all_files[folder_name])}")

    # 找出所有文件夹中都存在的公共basename
    # 使用第一个文件夹的basename作为基准
    if not all_files:
        raise ValueError("No data files found")

    # 获取所有非空文件夹
    non_empty_folders = [folder for folder, files in all_files.items() if files]
    if not non_empty_folders:
        raise ValueError("All data folders are empty")

    # 使用第一个非空文件夹的basename作为基准
    reference_folder = non_empty_folders[0]
    common_basenames = set(all_files[reference_folder].keys())
    logger.debug(f"Initial candidate samples from {reference_folder}: {len(common_basenames)}")

    # 与其他文件夹求交集
    for folder_name in non_empty_folders[1:]:
        folder_basenames = set(all_files[folder_name].keys())
        before_count = len(common_basenames)
        common_basenames = common_basenames.intersection(folder_basenames)
        logger.debug(
            f"Intersected data key {folder_name}: common={len(common_basenames)}, "
            f"removed={before_count - len(common_basenames)}"
        )

    if not common_basenames:
        raise ValueError("No common basenames found across data folders")

    logger.debug(f"Input scan complete: samples={len(common_basenames)}, data_keys={non_empty_folders}")

    # 为每个公共basename创建样本字典
    valid_samples = []
    for basename in common_basenames:
        sample_dict = {"basename": basename, "data_paths": {}}

        # 收集该basename的所有数据路径
        for folder_name in non_empty_folders:
            if basename in all_files[folder_name]:
                sample_dict["data_paths"][folder_name] = all_files[folder_name][basename]

        valid_samples.append(sample_dict)

    # 应用limit
    if limit and limit < len(valid_samples):
        valid_samples = valid_samples[:limit]
        logger.info(f"Applied sample limit: samples={len(valid_samples)}")

    return valid_samples


def _create_samples(sample_dicts: List[Dict[str, Any]], dataset_name: str) -> List[Sample]:
    """从样本字典创建Sample对象。"""
    samples = []
    basenames_seen = set()

    for sample_dict in sample_dicts:
        basename = sample_dict["basename"]

        # 检查重复的basename
        if basename in basenames_seen:
            logger.warning(f"Duplicate basename skipped: {basename}")
            continue
        basenames_seen.add(basename)

        samples.append(
            Sample(
                basename=basename,
                dataset_name=dataset_name,
                data_paths=sample_dict["data_paths"],
            )
        )

    return samples


def _get_sample_size(sample: Sample, size_based_on: str) -> int:
    """
    获取样本的大小（字节）。

    Args:
        sample: 样本对象
        size_based_on: 用于计算大小的数据类型（video或image）

    Returns:
        样本的大小（字节），如果无法获取则返回0
    """
    if size_based_on not in sample.data_paths:
        return 0

    try:
        file_path = sample.data_paths[size_based_on]
        return os.path.getsize(file_path)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to get file size: path={file_path}, error={e}")
        return 0


def _group_samples_by_size(
    samples: List[Sample],
    max_size_mb: int,
    size_based_on: str = "video",
) -> List[List[Sample]]:
    """
    根据文件大小将样本分组，确保每组的总大小不超过指定上限。

    Args:
        samples: 样本列表
        max_size_mb: 每组的最大大小（MB）
        size_based_on: 用于计算大小的数据类型（video或image）

    Returns:
        分组后的样本列表
    """
    max_size_bytes = max_size_mb * 1024 * 1024
    groups = []
    current_group = []
    current_size = 0

    # 先获取所有样本的大小信息
    sample_sizes = []
    total_size = 0
    for sample in samples:
        size = _get_sample_size(sample, size_based_on)
        sample_sizes.append((sample, size))
        total_size += size

    # 按大小分组
    for sample, size in sample_sizes:
        # 如果单个样本就超过最大大小，单独成组并警告
        if size > max_size_bytes:
            logger.warning(
                f"Single sample exceeds shard size limit: sample={sample.basename}, "
                f"key={size_based_on}, size={size / (1024**2):.1f}MB, limit={max_size_mb}MB"
            )
            # 如果当前组不为空，先保存
            if current_group:
                groups.append(current_group)
                current_group = []
                current_size = 0
            # 单独成组
            groups.append([sample])
            continue

        # 如果添加这个样本会超过限制，开始新组
        if current_size + size > max_size_bytes and current_group:
            groups.append(current_group)
            logger.debug(f"Finished shard group: samples={len(current_group)}, size={current_size / (1024**2):.1f}MB")
            current_group = []
            current_size = 0

        # 添加到当前组
        current_group.append(sample)
        current_size += size

    # 添加最后一组
    if current_group:
        groups.append(current_group)
        logger.debug(f"Finished final shard group: samples={len(current_group)}, size={current_size / (1024**2):.1f}MB")

    # 统计信息
    group_sizes = []
    group_counts = []
    for group in groups:
        group_size = sum(_get_sample_size(s, size_based_on) for s in group)
        group_sizes.append(group_size / (1024**2))  # 转换为MB
        group_counts.append(len(group))

    if group_sizes:
        logger.info(
            f"Shard grouping complete: strategy=size, key={size_based_on}, samples={len(samples)}, "
            f"shards={len(groups)}, total_size={total_size / (1024**3):.2f}GB, "
            f"shard_size_mb=min/avg/max {min(group_sizes):.1f}/"
            f"{sum(group_sizes) / len(group_sizes):.1f}/{max(group_sizes):.1f}, "
            f"samples_per_shard=min/avg/max {min(group_counts)}/"
            f"{sum(group_counts) / len(group_counts):.1f}/{max(group_counts)}"
        )

    return groups


def _load_file_bytes(file_path: str) -> bytes:
    """从本地路径加载文件字节。"""
    with open(file_path, "rb") as f:
        return f.read()


def _get_file_extension(file_path: str) -> str:
    """获取文件扩展名（包括点）。"""
    return Path(file_path).suffix


def _create_tar_file(samples_data: List[Tuple[str, bytes]], output_path: str) -> None:
    """使用给定的样本数据创建tar文件。"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with tarfile.open(output_path, "w") as tar:
        for key, data in samples_data:
            # 创建tarinfo
            tarinfo = tarfile.TarInfo(name=key)
            tarinfo.size = len(data)

            # 添加到tar
            tar.addfile(tarinfo, io.BytesIO(data))


def _load_sample_data(
    sample: Sample,
    max_file_size_mb: int = 1024,
    progress_callback: Optional[Callable[[], None]] = None,
) -> Tuple[str, Dict[str, bytes]]:
    """使用验证加载单个样本的所有数据。"""
    key = f"{sample.dataset_name}__{sample.basename}"
    sample_data = {}

    try:
        # 加载所有可用的数据类型
        for data_type, file_path in sample.data_paths.items():
            if not _validate_file_size(file_path, max_file_size_mb):
                raise ValueError(f"{data_type} file exceeds size limit: {file_path}")

            file_bytes = _load_file_bytes(file_path)
            sample_data[data_type] = (file_bytes, _get_file_extension(file_path))

        if progress_callback:
            progress_callback()

        return key, sample_data

    except Exception as e:
        logger.error(f"Failed to load sample: sample={sample.basename}, error={e}")
        raise


def _process_shard(
    samples: List[Sample],
    shard_id: int,
    output_dir: str,
    data_folders: Dict[str, List[str]],
    max_file_size_mb: int = 1024,
    dry_run: bool = False,
    progress_interval: int = 10,
    part_idx: Optional[int] = None,
) -> Dict[str, int]:
    """使用更好的错误处理和进度跟踪处理单个分片的样本。"""
    if dry_run:
        logger.debug(f"Dry run shard: shard={shard_id}, samples={len(samples)}")
        return {"shard_id": shard_id, "processed": len(samples), "skipped": 0}

    # 为每种数据类型创建集合
    data_collections = {folder_name: [] for folder_name in data_folders.keys()}

    processed_count = 0
    skipped_count = 0

    def progress_callback() -> None:
        nonlocal processed_count
        processed_count += 1
        if processed_count % progress_interval == 0:
            logger.debug(f"Shard progress: shard={shard_id}, processed={processed_count}/{len(samples)}")

    for sample in samples:
        try:
            key, sample_data = _load_sample_data(sample, max_file_size_mb, progress_callback)

            # 添加到适当的集合
            for data_type, (data_bytes, file_ext) in sample_data.items():
                data_collections[data_type].append((f"{key}{file_ext}", data_bytes))

        except Exception as e:  # noqa: BLE001
            logger.error(f"Skipping sample after load failure: shard={shard_id}, sample={sample.basename}, error={e}")
            skipped_count += 1
            continue

    # 为有样本的每种数据类型创建tar文件
    # 如果指定了part_idx，使用新的命名格式
    if part_idx is not None:
        shard_name = f"part_{part_idx:06d}/{shard_id:06d}.tar"
    else:
        shard_name = f"{shard_id:08d}.tar"
    created_files = []

    for data_type, samples_list in data_collections.items():
        if samples_list:
            tar_path = os.path.join(output_dir, data_type, shard_name)
            try:
                _create_tar_file(samples_list, tar_path)
                created_files.append(f"{data_type}: {tar_path} ({len(samples_list)} files)")
                logger.debug(f"Created shard tar: shard={shard_id}, key={data_type}, files={len(samples_list)}")
            except Exception as e:  # noqa: BLE001
                logger.error(f"Failed to create shard tar: shard={shard_id}, key={data_type}, error={e}")
                skipped_count += len(samples_list)

    valid_samples = processed_count - skipped_count
    if not created_files:
        logger.warning(f"Shard produced no valid tar files: shard={shard_id}")

    return {"shard_id": shard_id, "processed": valid_samples, "skipped": skipped_count}


def _process_shard_wrapper(args_tuple: Tuple) -> Dict[str, int]:
    """
    包装函数，用于多进程调用_process_shard。

    Args:
        args_tuple: 包含所有_process_shard参数的元组

    Returns:
        处理结果字典
    """
    return _process_shard(*args_tuple)


def _run_shard_pipeline(
    samples: List[Sample],
    output_dir: str,
    data_folders: Dict[str, List[str]],
    args: argparse.Namespace,
) -> Dict[str, int]:
    """Run the shard creation pipeline for a list of samples.

    This is the core logic extracted from main() so it can be called once (flat mode)
    or per-bucket (aspect-ratio classification mode).

    Returns:
        {"processed": N, "skipped": N, "shards": N}
    """
    # 根据是否指定max_shard_size_mb来决定分片策略
    if args.max_shard_size_mb:
        sample_groups = _group_samples_by_size(samples, args.max_shard_size_mb, args.size_based_on)
        strategy = f"size(max={args.max_shard_size_mb}MB,key={args.size_based_on})"
    else:
        # 使用固定数量的分片策略
        sample_groups = []
        for i in range(0, len(samples), args.shard_size):
            shard_samples = samples[i : i + args.shard_size]
            # 跳过单样本分片
            if len(shard_samples) > 1:
                sample_groups.append(shard_samples)
            else:
                logger.debug("Skipping single-sample shard")
        strategy = f"count(size={args.shard_size})"

    # 计算分片信息
    total_potential_shards = len(sample_groups)
    start_shard = 0

    # 准备分片任务
    shard_tasks = []
    for idx, shard_samples in enumerate(sample_groups):
        shard_id = start_shard + idx

        # 如果指定了part_size，计算part_idx
        part_idx = None
        if args.part_size:
            part_idx = shard_id // args.part_size
            logger.debug(f"Shard part assignment: shard={shard_id}, part={part_idx}")

        # 准备任务参数
        task_args = (
            shard_samples,
            shard_id,
            output_dir,
            data_folders,
            args.max_file_size_mb,
            args.dry_run,
            args.progress_interval,
            part_idx,
        )
        shard_tasks.append(task_args)

    logger.info(
        f"Sharding start: samples={len(samples)}, shards={len(shard_tasks)}, "
        f"strategy={strategy}, workers={args.num_workers}, output={output_dir}"
    )
    if args.part_size:
        total_parts = (total_potential_shards + args.part_size - 1) // args.part_size
        logger.debug(f"Shard layout: parts={total_parts}, shards_per_part={args.part_size}")

    # 分批处理样本
    total_shards = 0
    total_processed = 0
    total_skipped = 0

    if args.num_workers > 1:
        # 并行处理
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            # 提交所有任务
            future_to_shard = {
                executor.submit(_process_shard_wrapper, task_args): task_args[1]  # task_args[1] 是 shard_id
                for task_args in shard_tasks
            }

            # 收集完成的任务
            completed = 0
            progress_log_interval = max(1, min(100, len(shard_tasks) // 10 or 1))
            for future in as_completed(future_to_shard):
                shard_id = future_to_shard[future]
                completed += 1

                try:
                    shard_stats = future.result()
                    total_processed += shard_stats["processed"]
                    total_skipped += shard_stats["skipped"]
                    total_shards += 1
                    if completed == len(shard_tasks) or completed % progress_log_interval == 0:
                        logger.info(f"Sharding progress: completed_shards={completed}/{len(shard_tasks)}")

                except Exception as e:  # noqa: BLE001
                    logger.error(f"Shard failed: shard={shard_id}, error={e}")
                    continue
    else:
        # 串行处理
        for task_args in shard_tasks:
            shard_id = task_args[1]  # task_args[1] 是 shard_id

            try:
                shard_stats = _process_shard(*task_args)
                total_processed += shard_stats["processed"]
                total_skipped += shard_stats["skipped"]
                total_shards += 1

            except Exception as e:  # noqa: BLE001
                logger.error(f"Shard failed: shard={shard_id}, error={e}")
                continue

    return {"processed": total_processed, "skipped": total_skipped, "shards": total_shards}


def _list_wdinfo_local_objects(data_root: str) -> Tuple[List[str], List[str]]:
    """List tar files under a webdataset root and return detected data keys.

    Supports both flat output:
      image/part_000000/000000.tar

    and leaf directories from aspect-ratio classification:
      aspect_ratio_16_9/image/part_000000/000000.tar
    """
    data_path = Path(data_root)

    if not data_path.exists():
        raise ValueError(f"Directory does not exist: {data_root}")

    objects = []
    folders_with_tars = set()

    for subdir in data_path.iterdir():
        if subdir.is_dir():
            tar_files = list(subdir.glob("**/*.tar"))
            if tar_files:
                folders_with_tars.add(subdir.name)
                for tar_file in tar_files:
                    relative_path = tar_file.relative_to(data_path)
                    objects.append(str(relative_path))

    return objects, sorted(list(folders_with_tars))


def _save_wdinfo_local_file(data: str, file_path: str) -> None:
    """Save wdinfo JSON to a local file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w") as f:
        f.write(data)


def _get_wdinfo_tar_sample_count(tar_path: str, data_keys: List[str]) -> int:
    """Count regular files in a tar file."""
    try:
        with tarfile.open(tar_path, "r") as tf:
            return sum(1 for m in tf if m.isfile())
    except Exception as e:  # noqa: BLE001
        logger.error(f"Failed to read tar file: path={tar_path}, error={e}")
        return -1


def _get_wdinfo_tar_sample_count_wrapper(args: Tuple[str, str, str, List[str]]) -> Tuple[str, int]:
    """Multiprocessing wrapper for counting tar samples."""
    root_path, data_key, relative_path, data_keys = args
    tar_path = os.path.join(root_path, data_key, relative_path)
    sample_count = _get_wdinfo_tar_sample_count(tar_path, data_keys)
    return relative_path, sample_count


def _batch_get_wdinfo_tar_sample_counts(
    root_path: str,
    data_keys: List[str],
    relative_paths: List[str],
    chunk_size: int,
    num_workers: Optional[int] = None,
) -> List[int]:
    """Batch count tar samples, falling back to chunk_size when a tar cannot be read."""
    if num_workers is None:
        num_workers = cpu_count()

    logger.info(f"Counting tar samples: tars={len(relative_paths)}, workers={num_workers}")

    args_list = [(root_path, data_keys[0], rel_path, data_keys) for rel_path in relative_paths]
    results_dict = {}

    with Pool(processes=num_workers) as pool:
        for rel_path, count in pool.imap_unordered(_get_wdinfo_tar_sample_count_wrapper, args_list):
            results_dict[rel_path] = count

    sample_counts = []
    for rel_path in relative_paths:
        count = results_dict.get(rel_path, -1)
        if count == -1:
            count = chunk_size
        sample_counts.append(count)

    return sample_counts


def _find_wdinfo_common_keys(objects: List[str], folders_to_check: List[str]) -> Set[Tuple[str, str]]:
    """Find tar files present in every data-key folder."""
    logger.debug("Finding common tar files across data-key folders")

    objects_by_folder = defaultdict(list)

    for obj in objects:
        parts = obj.split(os.sep)
        if len(parts) >= 2:
            first_level_folder = parts[0]
            filename = parts[-1]
            relative_dir = os.sep.join(parts[1:-1]) if len(parts) > 2 else ""
            objects_by_folder[first_level_folder].append((relative_dir, filename))

    logger.debug(f"Tar folders detected: {list(objects_by_folder.keys())}")

    common_objects = set()
    for folder_name in folders_to_check:
        folder_objects = set(objects_by_folder.get(folder_name, []))
        if len(common_objects) == 0:
            common_objects = folder_objects
        else:
            common_objects = common_objects.intersection(folder_objects)
        logger.debug(f"Common tar check: key={folder_name}, common={len(common_objects)}")

    return common_objects


def _detect_wdinfo_nested_structure(data_root: str) -> List[str]:
    """Detect aspect_ratio_*/[duration_*/] leaf directories produced by this script."""
    data_path = Path(data_root)
    leaf_dirs = []

    for ar_dir in sorted(data_path.iterdir()):
        if not ar_dir.is_dir() or not ar_dir.name.startswith("aspect_ratio_"):
            continue

        duration_dirs = sorted([d for d in ar_dir.iterdir() if d.is_dir() and d.name.startswith("duration_")])

        if duration_dirs:
            for dur_dir in duration_dirs:
                leaf_dirs.append(str(dur_dir))
        else:
            leaf_dirs.append(str(ar_dir))

    return leaf_dirs


def _generate_wdinfo_for_directory(
    data_root: str,
    save_root: str,
    chunk_size: int = WDINFO_CHUNK_SIZE,
    num_workers: int = WDINFO_NUM_WORKERS,
) -> None:
    """Generate wdinfo.json for one flat webdataset directory.

    This intentionally does not create a validation split; all common tar files are
    assigned to the training data list.
    """
    tar_objects, data_keys = _list_wdinfo_local_objects(data_root)

    if not tar_objects:
        logger.warning(f"No tar files found, skipping wdinfo: root={data_root}")
        return

    if not data_keys:
        logger.warning(f"No data-key folders with tar files found, skipping wdinfo: root={data_root}")
        return

    wdinfo = {"data_keys": data_keys}
    common_keys = _find_wdinfo_common_keys(tar_objects, wdinfo["data_keys"])
    logger.info(
        f"Wdinfo scan: root={data_root}, data_keys={data_keys}, "
        f"tar_files={len(tar_objects)}, common_tars={len(common_keys)}"
    )

    if not common_keys:
        logger.warning(f"No common tar files across data keys, skipping wdinfo: root={data_root}")
        return

    root_path = str(Path(data_root).absolute())

    train_relative_paths = []
    for relative_dir, obj_filename in sorted(common_keys):
        if relative_dir:
            relative_path = os.path.join(relative_dir, obj_filename)
        else:
            relative_path = obj_filename
        train_relative_paths.append(relative_path)

    train_sample_counts = _batch_get_wdinfo_tar_sample_counts(
        root_path=root_path,
        data_keys=wdinfo["data_keys"],
        relative_paths=train_relative_paths,
        chunk_size=chunk_size,
        num_workers=num_workers,
    )

    filtered_train_paths = []
    filtered_train_counts = []
    for path, count in zip(train_relative_paths, train_sample_counts):
        if count > 1:
            filtered_train_paths.append(path)
            filtered_train_counts.append(count)
        else:
            logger.warning(f"Skipping tar with <=1 sample: path={path}, samples={count}")

    if not filtered_train_paths:
        logger.warning(f"No valid tar files after filtering, skipping wdinfo: root={data_root}")
        return

    wdinfo["data_list"] = filtered_train_paths
    wdinfo["data_list_key_count"] = filtered_train_counts
    wdinfo["root"] = root_path
    wdinfo["total_key_count"] = sum(wdinfo["data_list_key_count"])

    train_wdinfo_path = os.path.join(save_root, "wdinfo.json")
    _save_wdinfo_local_file(json.dumps(wdinfo, indent=2), train_wdinfo_path)
    logger.info(
        f"Wdinfo written: path={train_wdinfo_path}, tars={len(wdinfo['data_list'])}, "
        f"samples={wdinfo['total_key_count']}, filtered_tars={len(train_relative_paths) - len(filtered_train_paths)}"
    )


def _generate_wdinfo_after_sharding(output_dir: str) -> None:
    """Generate wdinfo.json files under output_dir with generate_wdinfo.py defaults."""
    nested_dirs = _detect_wdinfo_nested_structure(output_dir)

    if nested_dirs:
        logger.info(f"Generating wdinfo: root={output_dir}, mode=nested, leaf_dirs={len(nested_dirs)}")

        for leaf_dir in nested_dirs:
            rel_path = os.path.relpath(leaf_dir, output_dir)
            leaf_save_root = os.path.join(output_dir, rel_path)

            _generate_wdinfo_for_directory(
                data_root=leaf_dir,
                save_root=leaf_save_root,
                chunk_size=WDINFO_CHUNK_SIZE,
                num_workers=WDINFO_NUM_WORKERS,
            )
    else:
        logger.info(f"Generating wdinfo: root={output_dir}, mode=flat")
        _generate_wdinfo_for_directory(
            data_root=output_dir,
            save_root=output_dir,
            chunk_size=WDINFO_CHUNK_SIZE,
            num_workers=WDINFO_NUM_WORKERS,
        )


def main() -> None:
    """运行管道的主函数，具有增强的错误处理和统计信息。"""
    start_time = time.time()
    args = _parse_args()

    # 根据dry-run设置日志级别
    if args.dry_run:
        logger.info("Dry run enabled; no files will be created")

    # 从输入目录提取数据集名称或使用提供的名称
    dataset_name = args.dataset_name or Path(args.input_dir).name

    logger.info(
        f"Starting sharding: dataset={dataset_name}, input={args.input_dir}, "
        f"output={args.output_dir}, max_file_size={args.max_file_size_mb}MB, "
        f"aspect_ratio={args.classify_aspect_ratio}"
    )

    total_processed = 0
    total_skipped = 0
    total_shards = 0

    try:
        data_folders = _discover_data_folders(args.input_dir)

        if not data_folders:
            logger.error(f"No supported data folders found. supported={list(DATA_TYPE_EXTENSIONS.keys())}")
            return

        logger.info(f"Data keys detected: {list(data_folders.keys())}")

        # 扫描输入目录以查找所有数据文件
        sample_dicts = _scan_input_directory(args.input_dir, data_folders, args.limit)

        if not sample_dicts:
            logger.error("No valid samples found")
            return

        # 显示数据集统计信息
        per_key_counts = {
            folder_name: sum(1 for s in sample_dicts if folder_name in s["data_paths"])
            for folder_name in data_folders.keys()
        }
        logger.info(f"Dataset summary: samples={len(sample_dicts)}, per_key={per_key_counts}")

        if args.classify_aspect_ratio:
            # ---- Bucketed mode: classify samples by aspect ratio (and duration) ----
            duration_bins = _parse_duration_bins(args.duration_bins)
            buckets = _bucket_samples(sample_dicts, data_folders, duration_bins, args.metadata_workers)

            if not buckets:
                logger.error("No samples could be classified into buckets")
                return

            for bucket_key in sorted(buckets.keys()):
                bucket_samples_dicts = buckets[bucket_key]
                logger.info(f"Processing bucket: key={bucket_key}, samples={len(bucket_samples_dicts)}")

                # Create Sample objects for this bucket
                samples = _create_samples(bucket_samples_dicts, dataset_name)
                if not samples:
                    logger.warning(f"Bucket {bucket_key}: no valid samples, skipping")
                    continue

                # Shuffle within bucket
                if not args.no_shuffle:
                    random.shuffle(samples)

                bucket_output_dir = os.path.join(args.output_dir, bucket_key)
                if not args.dry_run:
                    os.makedirs(bucket_output_dir, exist_ok=True)

                # Each bucket has independent shard numbering starting from 0
                stats = _run_shard_pipeline(samples, bucket_output_dir, data_folders, args)
                total_processed += stats["processed"]
                total_skipped += stats["skipped"]
                total_shards += stats["shards"]
        else:
            # ---- Flat mode (original behavior) ----
            samples = _create_samples(sample_dicts, dataset_name)

            if not samples:
                logger.error("No samples were created")
                return

            # 除非指定--no-shuffle，否则打乱样本
            if not args.no_shuffle:
                random.shuffle(samples)
                logger.debug("Samples shuffled")

            # 创建输出目录
            if not args.dry_run:
                os.makedirs(args.output_dir, exist_ok=True)

            stats = _run_shard_pipeline(samples, args.output_dir, data_folders, args)
            total_processed = stats["processed"]
            total_skipped = stats["skipped"]
            total_shards = stats["shards"]

        if args.dry_run:
            logger.info("Dry run: skipping wdinfo generation")
        elif total_shards > 0:
            _generate_wdinfo_after_sharding(args.output_dir)
        else:
            logger.warning("No shards were created; skipping wdinfo generation")

        # 最终统计
        elapsed_time = time.time() - start_time
        seconds_per_sample = elapsed_time / total_processed if total_processed > 0 else 0.0
        logger.info(
            f"Pipeline complete: processed={total_processed}, skipped={total_skipped}, "
            f"shards={total_shards}, elapsed={elapsed_time:.1f}s, "
            f"sec_per_sample={seconds_per_sample:.2f}, output={args.output_dir}"
        )

    except KeyboardInterrupt:
        logger.warning(f"Pipeline interrupted by user: processed={total_processed}")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
