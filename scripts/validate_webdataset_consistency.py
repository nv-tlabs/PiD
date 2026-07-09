#!/usr/bin/env python3
"""
Validate WebDataset tar consistency across data type folders.

For a given dataset root (or aspect_ratio leaf), verifies that all data type
folders (image_256, wan_latent_512, caption, umt5, ...) contain matching tar
files with identical basenames (ignoring extensions) in the same order.

Optimization: only scan the first data type folder to discover tar paths,
then assume the same structure for all other folders (and report missing tars).

Usage:
    # Validate a specific aspect_ratio leaf:
    python scripts/validate_webdataset_consistency.py \
        /path/to/dataset/aspect_ratio_1_1

    # Validate all aspect_ratio_* under a dataset root:
    python scripts/validate_webdataset_consistency.py \
        /path/to/dataset

    # Use more workers:
    python scripts/validate_webdataset_consistency.py \
        /path/to/dataset --num-workers 32
"""

import argparse
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed


def get_basenames_from_tar(tar_path: str) -> list[str]:
    """Extract ordered basenames (without extension) from a tar file using bash tar."""
    result = subprocess.run(
        ["tar", "-tf", tar_path],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"tar -tf failed: {result.stderr.strip()}")
    names = [line for line in result.stdout.splitlines() if line]
    return [os.path.splitext(n)[0] for n in names]


def find_all_tars(folder: str) -> list[str]:
    """Find all tar files under a folder, sorted by relative path."""
    tars = []
    for root, dirs, files in os.walk(folder):
        dirs.sort()
        for f in files:
            if f.endswith(".tar"):
                tars.append(os.path.relpath(os.path.join(root, f), folder))
    tars.sort()
    return tars


def validate_single_tar(args: tuple) -> dict:
    """Validate a single tar's basenames across all data type folders.

    Only reads the reference tar fully; for other types, compares basenames.
    args: (rel_tar, leaf_dir, data_types)
    """
    rel_tar, leaf_dir, data_types = args
    result = {"rel_tar": rel_tar, "ok": True, "errors": []}

    ref_type = data_types[0]
    ref_path = os.path.join(leaf_dir, ref_type, rel_tar)

    try:
        ref_basenames = get_basenames_from_tar(ref_path)
    except Exception as e:
        result["ok"] = False
        result["errors"].append(f"  [{ref_type}] Failed to read reference tar: {e}")
        return result

    for dtype in data_types[1:]:
        full_path = os.path.join(leaf_dir, dtype, rel_tar)

        if not os.path.exists(full_path):
            result["ok"] = False
            result["errors"].append(f"  [{dtype}] MISSING tar: {rel_tar}")
            continue

        try:
            basenames = get_basenames_from_tar(full_path)
        except Exception as e:
            result["ok"] = False
            result["errors"].append(f"  [{dtype}] Failed to read tar: {e}")
            continue

        if len(basenames) != len(ref_basenames):
            result["ok"] = False
            result["errors"].append(
                f"  [{dtype}] count mismatch vs [{ref_type}]: {len(basenames)} vs {len(ref_basenames)}"
            )
            continue

        mismatches = []
        for i, (a, b) in enumerate(zip(ref_basenames, basenames)):
            if a != b:
                mismatches.append(f"    idx {i}: {ref_type}={a!r}  vs  {dtype}={b!r}")
                if len(mismatches) >= 5:
                    mismatches.append(f"    ... (truncated, total {len(basenames)} entries)")
                    break
        if mismatches:
            result["ok"] = False
            result["errors"].append(f"  [{dtype}] basename mismatch vs [{ref_type}]:")
            result["errors"].extend(mismatches)

    return result


def validate_leaf(leaf_dir: str, num_workers: int = 16) -> tuple[bool, list[str]]:
    """Validate one leaf directory (e.g. an aspect_ratio_X_Y folder)."""
    logs = []
    leaf_dir = os.path.abspath(leaf_dir)
    logs.append(f"\n{'=' * 80}")
    logs.append(f"Validating: {leaf_dir}")
    logs.append(f"{'=' * 80}")

    data_types = sorted(d for d in os.listdir(leaf_dir) if os.path.isdir(os.path.join(leaf_dir, d)))

    if len(data_types) < 2:
        logs.append(f"  SKIP: only {len(data_types)} data type folder(s) found: {data_types}")
        return True, logs

    logs.append(f"  Data type folders ({len(data_types)}): {data_types}")

    # Only scan the first data type folder to discover tar paths
    ref_type = data_types[0]
    ref_path = os.path.join(leaf_dir, ref_type)
    print(f"  Scanning tar list from [{ref_type}] ...", flush=True)
    t0 = time.time()
    rel_tars = find_all_tars(ref_path)
    print(f"  Found {len(rel_tars)} tar(s) in {time.time() - t0:.1f}s", flush=True)

    logs.append(f"  Reference folder [{ref_type}]: {len(rel_tars)} tar(s)")

    # Build tasks — each task validates one tar across all data types
    tasks = [(rel_tar, leaf_dir, data_types) for rel_tar in rel_tars]

    logs.append(f"  Checking {len(tasks)} tar(s) × {len(data_types)} data types with {num_workers} workers ...")
    print(f"  Checking {len(tasks)} tar(s) × {len(data_types)} data types ...", flush=True)

    all_ok = True
    fail_count = 0
    done_count = 0
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(validate_single_tar, t): t for t in tasks}
        for future in as_completed(futures):
            done_count += 1
            if done_count % 100 == 0 or done_count == len(tasks):
                elapsed = time.time() - t0
                rate = done_count / elapsed if elapsed > 0 else 0
                print(
                    f"\r  Progress: {done_count}/{len(tasks)} "
                    f"({100 * done_count / len(tasks):.0f}%) "
                    f"[{elapsed:.0f}s, {rate:.1f} tar/s, {fail_count} fails]",
                    end="",
                    flush=True,
                )

            result = future.result()
            if not result["ok"]:
                all_ok = False
                fail_count += 1
                if fail_count <= 20:
                    logs.append(f"  FAIL {result['rel_tar']}:")
                    logs.extend(result["errors"])
                elif fail_count == 21:
                    logs.append("  ... (additional failures truncated)")

    print(flush=True)  # newline after progress

    elapsed = time.time() - t0
    if all_ok:
        logs.append(f"  PASS: all {len(tasks)} tar(s) consistent across {len(data_types)} data types ({elapsed:.0f}s)")
    else:
        logs.append(f"  FAILED: {fail_count}/{len(tasks)} tar(s) have inconsistencies ({elapsed:.0f}s)")

    return all_ok, logs


def main():
    parser = argparse.ArgumentParser(description="Validate WebDataset tar consistency across data type folders.")
    parser.add_argument(
        "data_root",
        help="Path to a dataset root (containing aspect_ratio_* dirs) or a single leaf dir.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=16,
        help="Number of parallel workers for tar validation (default: 16).",
    )
    args = parser.parse_args()

    data_root = os.path.abspath(args.data_root)

    # Determine leaf directories to validate
    leaves = []
    children = sorted(os.listdir(data_root))
    aspect_dirs = [d for d in children if d.startswith("aspect_ratio_")]

    if aspect_dirs:
        for ad in aspect_dirs:
            ad_path = os.path.join(data_root, ad)
            sub_children = sorted(os.listdir(ad_path))
            duration_dirs = [d for d in sub_children if d.startswith("duration_")]
            if duration_dirs:
                for dd in duration_dirs:
                    leaves.append(os.path.join(ad_path, dd))
            else:
                leaves.append(ad_path)
    else:
        leaves.append(data_root)

    print(f"Found {len(leaves)} leaf dir(s) to validate.\n")

    all_ok = True
    for leaf in leaves:
        ok, logs = validate_leaf(leaf, num_workers=args.num_workers)
        for line in logs:
            print(line)
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print("ALL PASSED")
    else:
        print("SOME CHECKS FAILED")
        exit(1)


if __name__ == "__main__":
    main()
