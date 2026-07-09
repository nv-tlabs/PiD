"""
Download MultiAspect-4K-1M image URLs and write per-image caption JSON files.

Input JSON files are expected to contain either a list of image records or one image
record. Each record should have `image_url` and an English caption field such as
`en_caption`. For every valid record, this script writes:

- image/<source_json_stem>_<record_index>.<image_ext>
- caption/<source_json_stem>_<record_index>.json

The caption JSON format matches the webdataset sharding script:
{"prompt": str, "prompt_medium": str, "prompt_short": str}
"""

import argparse
import json
import os
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse

from tqdm import tqdm

DEFAULT_INPUT_DIR = "raw_data/MultiAspect-4K-1M"
DEFAULT_OUTPUT_DIR = "raw_data/MultiAspect-4K-1M-download"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
USER_AGENT = "Mozilla/5.0 (compatible; linearvsr-multiaspect-downloader/1.0)"


@dataclass(frozen=True)
class DownloadTask:
    uid: str
    url: str
    caption: str
    image_path: Path
    caption_path: Path
    source_json: str
    source_index: int


@dataclass(frozen=True)
class DownloadResult:
    status: str
    uid: str
    image_path: str
    caption_path: str
    source_json: str
    source_index: int
    error: Optional[str] = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download MultiAspect-4K-1M images and create caption JSON files.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(DEFAULT_INPUT_DIR),
        help=f"Directory containing source JSON files. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help=f"Output directory. Images go to image/, captions go to caption/. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--max-json-files",
        type=int,
        default=None,
        help="Only process the first N JSON files after sorting by filename. Default: all JSON files.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Only process the first N valid image records after JSON-file filtering. Default: no limit.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Number of concurrent download threads. Default: 32.",
    )
    parser.add_argument(
        "--max-pending",
        type=int,
        default=None,
        help="Maximum queued futures. Default: workers * 4.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="HTTP timeout in seconds for each request. Default: 30.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Number of download attempts per image. Default: 3.",
    )
    parser.add_argument(
        "--retry-sleep",
        type=float,
        default=1.0,
        help="Initial retry sleep in seconds. Backoff doubles after each failed attempt. Default: 1.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload existing images and rewrite existing caption files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan input JSON files and print the planned output paths without downloading.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=1,
        help="Refresh the progress bar every N completed tasks. Set to 0 for automatic refresh. Default: 1.",
    )
    return parser.parse_args()


def _list_json_files(input_dir: Path, max_json_files: Optional[int]) -> List[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    json_files = sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix == ".json")
    if max_json_files is not None:
        if max_json_files < 0:
            raise ValueError("--max-json-files must be non-negative")
        json_files = json_files[:max_json_files]
    return json_files


def _load_records(json_path: Path) -> List[Dict[str, Any]]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        if isinstance(data.get("data"), list):
            records = data["data"]
        elif isinstance(data.get("items"), list):
            records = data["items"]
        else:
            records = [data]
    else:
        raise ValueError(f"Unsupported JSON structure in {json_path}: {type(data).__name__}")

    valid_records = []
    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            print(f"[warn] Skip non-dict record: json={json_path} index={idx}")
            continue
        valid_records.append(record)
    return valid_records


def _caption_from_record(record: Dict[str, Any]) -> Optional[str]:
    for key in ("en_caption", "caption", "prompt", "prompt_medium", "prompt_short"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _infer_image_extension(url: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return suffix
    return ".jpg"


def _iter_download_tasks(
    json_files: Iterable[Path],
    output_dir: Path,
    max_items: Optional[int],
) -> Iterable[DownloadTask]:
    image_dir = output_dir / "image"
    caption_dir = output_dir / "caption"
    emitted = 0

    for json_path in json_files:
        try:
            records = _load_records(json_path)
        except Exception as e:
            print(f"[warn] Skip unreadable JSON file: json={json_path} error={e}")
            continue

        use_plain_stem = len(records) == 1
        for idx, record in enumerate(records):
            if max_items is not None and emitted >= max_items:
                return

            url = record.get("image_url")
            caption = _caption_from_record(record)
            if not isinstance(url, str) or not url.strip():
                print(f"[warn] Skip record without image_url: json={json_path} index={idx}")
                continue
            if caption is None:
                print(f"[warn] Skip record without English caption: json={json_path} index={idx}")
                continue

            uid = json_path.stem if use_plain_stem else f"{json_path.stem}_{idx:06d}"
            image_path = image_dir / f"{uid}{_infer_image_extension(url)}"
            caption_path = caption_dir / f"{uid}.json"
            emitted += 1
            yield DownloadTask(
                uid=uid,
                url=url.strip(),
                caption=caption,
                image_path=image_path,
                caption_path=caption_path,
                source_json=str(json_path),
                source_index=idx,
            )


def _is_nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _write_caption(caption_path: Path, caption: str) -> None:
    payload = {
        "prompt": caption,
        "prompt_medium": caption,
        "prompt_short": caption,
    }
    tmp_path = caption_path.with_name(f"{caption_path.name}.tmp.{os.getpid()}.{threading.get_ident()}")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp_path, caption_path)


def _download_url(url: str, image_path: Path, timeout: float) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    tmp_path = image_path.with_name(f"{image_path.name}.tmp.{os.getpid()}.{threading.get_ident()}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content_type = response.headers.get("Content-Type", "").split(";")[0].strip().lower()
            if content_type and not content_type.startswith("image/"):
                raise RuntimeError(f"unexpected content-type: {content_type}")

            with tmp_path.open("wb") as f:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)

        if not _is_nonempty_file(tmp_path):
            raise RuntimeError("downloaded file is empty")
        os.replace(tmp_path, image_path)
    except Exception:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _download_one(
    task: DownloadTask, timeout: float, retries: int, retry_sleep: float, overwrite: bool
) -> DownloadResult:
    if not overwrite and _is_nonempty_file(task.image_path):
        if overwrite or not task.caption_path.is_file():
            _write_caption(task.caption_path, task.caption)
            status = "caption_written"
        else:
            status = "skipped"
        return DownloadResult(
            status=status,
            uid=task.uid,
            image_path=str(task.image_path),
            caption_path=str(task.caption_path),
            source_json=task.source_json,
            source_index=task.source_index,
        )

    attempts = max(1, retries)
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            _download_url(task.url, task.image_path, timeout)
            _write_caption(task.caption_path, task.caption)
            return DownloadResult(
                status="downloaded",
                uid=task.uid,
                image_path=str(task.image_path),
                caption_path=str(task.caption_path),
                source_json=task.source_json,
                source_index=task.source_index,
            )
        except (urllib.error.URLError, OSError, RuntimeError) as e:
            last_error = str(e)
            if attempt < attempts:
                time.sleep(retry_sleep * (2 ** (attempt - 1)))

    return DownloadResult(
        status="failed",
        uid=task.uid,
        image_path=str(task.image_path),
        caption_path=str(task.caption_path),
        source_json=task.source_json,
        source_index=task.source_index,
        error=last_error,
    )


def _print_dry_run(tasks: Iterable[DownloadTask], limit: int = 10) -> None:
    count = 0
    for count, task in enumerate(tasks, start=1):
        if count <= limit:
            print(f"[dry-run] uid={task.uid} image={task.image_path} caption={task.caption_path} url={task.url}")
    print(f"[dry-run] valid image records: {count}")


def _run_downloads(args: argparse.Namespace, tasks: Iterable[DownloadTask], total_tasks: int) -> Counter:
    stats: Counter = Counter()
    max_pending = args.max_pending or args.workers * 4
    futures = set()
    task_iter = iter(tasks)
    miniters = args.progress_interval if args.progress_interval > 0 else None

    progress = tqdm(
        total=total_tasks,
        desc="Downloading",
        unit="image",
        miniters=miniters,
        dynamic_ncols=True,
    )

    def submit_until_full(executor: ThreadPoolExecutor) -> None:
        while len(futures) < max_pending:
            try:
                task = next(task_iter)
            except StopIteration:
                return
            future = executor.submit(_download_one, task, args.timeout, args.retries, args.retry_sleep, args.overwrite)
            futures.add(future)
            stats["submitted"] += 1

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            submit_until_full(executor)
            while futures:
                done, futures_left = wait(futures, return_when=FIRST_COMPLETED)
                futures = futures_left
                for future in done:
                    result = future.result()
                    stats[result.status] += 1
                    stats["completed"] += 1
                    if result.status == "failed":
                        tqdm.write(
                            "[failed] "
                            f"uid={result.uid} json={result.source_json} "
                            f"index={result.source_index} error={result.error}"
                        )
                progress.set_postfix(
                    downloaded=stats["downloaded"],
                    skipped=stats["skipped"],
                    caption_written=stats["caption_written"],
                    failed=stats["failed"],
                    refresh=False,
                )
                progress.update(len(done))
                submit_until_full(executor)
    finally:
        progress.close()

    return stats


def main() -> None:
    args = _parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.max_pending is not None and args.max_pending < args.workers:
        raise ValueError("--max-pending must be greater than or equal to --workers")
    if args.max_items is not None and args.max_items < 0:
        raise ValueError("--max-items must be non-negative")

    json_files = _list_json_files(args.input_dir, args.max_json_files)
    print(f"[info] input_dir={args.input_dir}")
    print(f"[info] output_dir={args.output_dir}")
    print(f"[info] json_files={len(json_files)} workers={args.workers}")

    tasks = _iter_download_tasks(json_files, args.output_dir, args.max_items)
    if args.dry_run:
        _print_dry_run(tasks)
        return

    print("[info] scanning valid image records...")
    total_tasks = sum(1 for _ in tasks)
    print(f"[info] total_images={total_tasks}")

    (args.output_dir / "image").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "caption").mkdir(parents=True, exist_ok=True)
    tasks = _iter_download_tasks(json_files, args.output_dir, args.max_items)
    stats = _run_downloads(args, tasks, total_tasks)
    print(
        "[summary] "
        f"submitted={stats['submitted']} completed={stats['completed']} "
        f"downloaded={stats['downloaded']} skipped={stats['skipped']} "
        f"caption_written={stats['caption_written']} failed={stats['failed']}"
    )


if __name__ == "__main__":
    main()
