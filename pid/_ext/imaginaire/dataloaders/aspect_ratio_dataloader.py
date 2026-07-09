# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""AspectRatioDataLoader: A dataloader wrapper that batches samples by aspect ratio.

Design:
  - A single FIFO queue preserves the original fetch order from the webdataset,
    so the output AR distribution matches the dataset's natural distribution.
  - For batch_size=1 the FIFO is consumed directly (pure pass-through).
  - For batch_size>1, samples are grouped by AR.  The AR whose oldest sample
    arrived first is batched next, keeping output order as close to fetch order
    as the same-shape constraint allows.

Key invariants / fixes vs. the previous implementation:
  1. yield happens OUTSIDE the lock and OUTSIDE watchdog timers. The prefetch
     thread is never blocked by the training step, and downstream
     training/communication stalls are not attributed to dataloader batching
     (Fix 1).
  2. ResizeScale upscale is handled upstream (Fix 2, in resize.py).
  3. No silent sample drops — when the FIFO or per-AR index is full the
     prefetch thread waits instead of spinning (Fix 3).
  4. decoder_handler=warn_and_continue is set upstream (in dataset_provider.py).
  5. If the global FIFO is full but no aspect-ratio bucket has enough samples
     for a full batch, emit a partial batch from the largest bucket. This avoids
     a producer/consumer deadlock for high batch_size multi-AR image loaders.
"""

import threading
import traceback
from collections import defaultdict, deque
from typing import ClassVar, Dict, Iterator, List, Optional

import torch
from torch.utils.data import DataLoader

from pid._ext.imaginaire.datasets.watchdog import OperationWatchdog
from pid._ext.imaginaire.utils import log


class AspectRatioDataLoader:
    """A DataLoader wrapper that batches samples by aspect ratio.

    Internally a single FIFO holds all prefetched samples in arrival order.
    A secondary index (per-AR deques of FIFO positions) enables efficient
    same-AR batch formation for batch_size > 1.

    Attributes:
        data_loader (DataLoader): The underlying dataloader (batch_size=1).
        batch_size (int): Target batch size for yielded batches.
        max_queue_size (int): Maximum total samples buffered.
    """

    def __init__(
        self,
        data_loader: DataLoader,
        batch_size: int,
        max_queue_size_per_ar: Optional[int] = None,
        total_max_samples: Optional[int] = None,
        name: str = "aspect_ratio_dataloader",
    ) -> None:
        self.data_loader = data_loader
        self.batch_size = batch_size
        # Total buffer capacity.  Per-AR limit is no longer needed — the single
        # FIFO with a global cap prevents both memory blow-up and the spin-drop
        # bug that the old per-AR limit caused.
        self.max_queue_size = total_max_samples or batch_size * 50
        self.name = name

        # Iterator over the underlying DataLoader. Start it lazily on the first
        # __iter__ call; joint video+image training otherwise starts both
        # streams' workers immediately and creates large object-store bursts.
        self._data_iter: Optional[Iterator] = None

        # --- shared state protected by _cond ---
        self._cond = threading.Condition()  # guards all fields below
        self._fifo: deque = deque()  # FIFO of (sample_dict, aspect_ratio)
        self._ar_indices: Dict[str, deque] = defaultdict(deque)  # AR -> deque of FIFO indices
        self._next_seq: int = 0  # monotonic counter for FIFO ordering
        self._data_exhausted: bool = False
        self._prefetch_exception: Optional[Exception] = None

        # Stop signal
        self._stop_event = threading.Event()

        # Monitoring
        self._watchdog = OperationWatchdog(warning_threshold=100, verbose_interval=600, name=name)
        self._aspect_ratio_stats: Dict[str, int] = defaultdict(int)
        self._samples_fetched: int = 0
        self._warned_unknown_ar: bool = False
        self._partial_batch_count: int = 0

        self._prefetch_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Prefetch thread
    # ------------------------------------------------------------------

    def _ensure_started(self) -> None:
        if self._prefetch_thread is not None:
            return
        self._data_iter = iter(self.data_loader)
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_loop, daemon=True, name=f"{self.name}_prefetch_thread"
        )
        self._prefetch_thread.start()

    def _prefetch_loop(self) -> None:
        """Continuously fetch samples and append to the FIFO.

        Waits when the FIFO reaches max_queue_size (backpressure).
        """
        try:
            while not self._stop_event.is_set():
                # Backpressure: wait until there is room in the FIFO.
                with self._cond:
                    while len(self._fifo) >= self.max_queue_size and not self._stop_event.is_set():
                        self._cond.wait(timeout=1.0)

                if self._stop_event.is_set():
                    break

                # Fetch next sample (outside the lock — may block on IO).
                try:
                    assert self._data_iter is not None
                    with self._watchdog.watch("fetch_sample", verbose_first_n=5):
                        sample = next(self._data_iter)
                except StopIteration:
                    with self._cond:
                        self._data_exhausted = True
                        self._cond.notify_all()
                    break
                except Exception as e:
                    self._set_exception(e, "Error fetching sample from DataLoader")
                    break

                # Unbatch (underlying DataLoader has batch_size=1).
                sample = self._unbatch(sample)

                ar = sample.get("aspect_ratio", "unknown")
                if ar == "unknown" and not self._warned_unknown_ar:
                    log.warning(f"[{self.name}] Sample missing 'aspect_ratio' key, using 'unknown'")
                    self._warned_unknown_ar = True
                sample["aspect_ratio"] = ar

                # Enqueue.
                with self._cond:
                    seq = self._next_seq
                    self._next_seq += 1
                    self._fifo.append((seq, ar, sample))
                    self._ar_indices[ar].append(seq)
                    self._aspect_ratio_stats[ar] += 1
                    self._samples_fetched += 1
                    if self._samples_fetched % 50000 == 0:
                        self._log_aspect_ratio_distribution()
                    self._cond.notify_all()

        except Exception as e:
            self._set_exception(e, "Unexpected error in prefetch thread")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unbatch(sample: dict) -> dict:
        """Remove the batch dimension added by the DataLoader (batch_size=1)."""
        if not isinstance(sample, dict):
            return sample
        out = {}
        for key, value in sample.items():
            if isinstance(value, torch.Tensor) and value.dim() > 0 and value.shape[0] == 1:
                out[key] = value.squeeze(0)
            elif isinstance(value, list) and len(value) == 1:
                out[key] = value[0]
            else:
                out[key] = value
        return out

    def _set_exception(self, exception: Exception, context: str = "") -> None:
        error_info = f"{context}: {str(exception)}\n{traceback.format_exc()}"
        with self._cond:
            self._prefetch_exception = RuntimeError(error_info)
            self._cond.notify_all()

    def _check_for_errors(self) -> None:
        if self._prefetch_exception is not None:
            raise self._prefetch_exception

    def _log_aspect_ratio_distribution(self) -> None:
        total = sum(self._aspect_ratio_stats.values())
        if total == 0:
            return
        log.info(f"[{self.name}] Aspect ratio distribution after {self._samples_fetched} samples:")
        for ar, count in sorted(self._aspect_ratio_stats.items()):
            pct = 100.0 * count / total
            log.info(f"[{self.name}]   '{ar}': {count}/{total} ({pct:.1f}%)")

    # String keys that are *batch-level metadata* (one value shared by every sample
    # in the batch by construction). For these we collapse to a single string when
    # all samples agree, mirroring how scalar metadata is typically consumed.
    # All OTHER string keys (e.g. caption / __url__ / __key__) are per-sample data
    # and must be returned as list[str] of length B — even when samples happen to
    # share the same value (e.g. all-arxiv batch where every caption is identical).
    # Returning a single str in that case silently downstreams a B=1 effective
    # batch and crashes the model on shape mismatch (see history of pixeldit_sr_model
    # caption broadcast / latent_noising._broadcast_urls).
    _SCALAR_STRING_KEYS: ClassVar[set[str]] = {"aspect_ratio"}

    @staticmethod
    def _collate_samples(samples: List[Dict]) -> Dict:
        """Collate a list of sample dicts into a single batched dict."""
        batch = {}
        all_keys = set()
        for s in samples:
            all_keys.update(s.keys())

        for key in all_keys:
            values = [s.get(key) for s in samples]
            values = [v for v in values if v is not None]
            if not values:
                continue
            first = values[0]
            if isinstance(first, str):
                if key in AspectRatioDataLoader._SCALAR_STRING_KEYS and all(v == first for v in values):
                    batch[key] = first
                else:
                    batch[key] = values
            elif isinstance(first, torch.Tensor):
                batch[key] = torch.stack(values, dim=0)
            elif isinstance(first, (int, float)):
                batch[key] = torch.tensor(values)
            elif isinstance(first, list):
                batch[key] = [item for v in values for item in v]
            else:
                batch[key] = values
        return batch

    # ------------------------------------------------------------------
    # FIFO helpers (must be called with self._cond held)
    # ------------------------------------------------------------------

    def _pop_front(self) -> Dict:
        """Pop the oldest sample from the FIFO (batch_size=1 fast path).

        Caller must hold self._cond.
        """
        seq, ar, sample = self._fifo.popleft()
        # Remove from AR index as well.
        ar_deque = self._ar_indices[ar]
        # The front of the AR deque must be this seq (FIFO order guarantee).
        if ar_deque and ar_deque[0] == seq:
            ar_deque.popleft()
        self._cond.notify_all()  # room freed → wake prefetch
        return sample

    def _find_oldest_ready_ar(self) -> Optional[str]:
        """Return the AR whose oldest sample arrived first among ARs with
        >= batch_size samples, or None.

        Caller must hold self._cond.
        """
        best_ar = None
        best_seq = None
        for ar, idx_deque in self._ar_indices.items():
            if len(idx_deque) >= self.batch_size:
                oldest = idx_deque[0]
                if best_seq is None or oldest < best_seq:
                    best_seq = oldest
                    best_ar = ar
        return best_ar

    def _find_largest_partial_ar(self) -> Optional[str]:
        """Return the largest non-empty AR bucket, oldest-first on ties.

        This is an emergency path used only when the global FIFO is full and no
        AR bucket can form a full batch. Without it, the consumer waits for a
        full same-AR batch while the producer waits for FIFO space.
        """
        best_ar = None
        best_count = 0
        best_seq = None
        for ar, idx_deque in self._ar_indices.items():
            count = len(idx_deque)
            if count == 0 or count >= self.batch_size:
                continue
            oldest = idx_deque[0]
            if count > best_count or (count == best_count and (best_seq is None or oldest < best_seq)):
                best_ar = ar
                best_count = count
                best_seq = oldest
        return best_ar

    def _pop_batch_for_ar(self, target_ar: str) -> List[Dict]:
        """Remove batch_size samples of *target_ar* from the FIFO.

        Removes them from both the FIFO and the AR index.
        Caller must hold self._cond.
        """
        # Collect the seq numbers to remove.
        ar_deque = self._ar_indices[target_ar]
        seqs_to_take = set()
        for _ in range(self.batch_size):
            seqs_to_take.add(ar_deque.popleft())

        # Walk the FIFO and extract matching entries.
        samples = []
        new_fifo = deque()
        for seq, ar, sample in self._fifo:
            if seq in seqs_to_take:
                samples.append(sample)
            else:
                new_fifo.append((seq, ar, sample))
        self._fifo = new_fifo
        self._cond.notify_all()  # room freed → wake prefetch
        return samples

    # ------------------------------------------------------------------
    # Iterator
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[Dict]:
        """Yield batches, preserving fetch-order AR distribution.

        - batch_size=1: pure FIFO pass-through.
        - batch_size>1: batches the AR whose oldest sample is globally oldest.

        IMPORTANT: yield happens OUTSIDE the lock and outside watchdog timers.
        A slow yield return means downstream training/communication is still
        working on the previous batch, not that this dataloader is forming a
        batch.
        """
        self._ensure_started()
        if self.batch_size == 1:
            yield from self._iter_bs1()
        else:
            yield from self._iter_bsN()

    def _iter_bs1(self) -> Iterator[Dict]:
        """Fast path for batch_size=1: pure FIFO, no AR grouping."""
        while not self._stop_event.is_set():
            batch = None
            exhausted = False

            with self._watchdog.watch(
                "wait_and_collate_batch",
                description=("wait for one queued sample and collate it; downstream training/yield time is excluded"),
                verbose_first_n=5,
            ):
                with self._cond:
                    # Wait for at least one sample.
                    while not self._fifo and not self._data_exhausted and not self._prefetch_exception:
                        self._cond.wait(timeout=1.0)

                    self._check_for_errors()

                    if self._data_exhausted and not self._fifo:
                        exhausted = True
                    elif self._fifo:
                        sample = self._pop_front()
                        # Collate single sample into a batch (adds batch dim to tensors).
                        batch = self._collate_samples([sample])

            if exhausted:
                break
            if batch is not None:
                yield batch

    def _iter_bsN(self) -> Iterator[Dict]:
        """Batching path for batch_size>1: group by AR, prefer oldest-first."""
        while not self._stop_event.is_set():
            batch = None
            exhausted = False
            partial_batches = None

            with self._watchdog.watch(
                "wait_and_collate_batch",
                description=(
                    "wait for enough same-aspect-ratio samples and collate them; "
                    "downstream training/yield time is excluded"
                ),
                verbose_first_n=5,
            ):
                with self._cond:
                    # Wait for any AR to have enough samples.
                    while (
                        self._find_oldest_ready_ar() is None
                        and len(self._fifo) < self.max_queue_size
                        and not self._data_exhausted
                        and not self._prefetch_exception
                    ):
                        self._cond.wait(timeout=1.0)

                    self._check_for_errors()

                    if self._data_exhausted:
                        # Drain remaining samples as partial batches per AR.
                        partial_batches = []
                        for ar in list(self._ar_indices.keys()):
                            if self._ar_indices[ar]:
                                samples = self._pop_batch_for_ar_partial(ar)
                                partial_batches.append(self._collate_samples(samples))
                        exhausted = True
                    else:
                        target_ar = self._find_oldest_ready_ar()
                        if target_ar is not None:
                            samples = self._pop_batch_for_ar(target_ar)
                            batch = self._collate_samples(samples)
                        elif len(self._fifo) >= self.max_queue_size:
                            target_ar = self._find_largest_partial_ar()
                            if target_ar is not None:
                                partial_size = len(self._ar_indices[target_ar])
                                self._partial_batch_count += 1
                                if self._partial_batch_count <= 10 or self._partial_batch_count % 100 == 0:
                                    log.warning(
                                        f"[{self.name}] FIFO full with no ready AR bucket; "
                                        f"yielding partial batch {partial_size}/{self.batch_size} "
                                        f"for aspect_ratio={target_ar!r} "
                                        f"(fifo={len(self._fifo)}/{self.max_queue_size}, "
                                        f"partial_count={self._partial_batch_count})"
                                    )
                                samples = self._pop_batch_for_ar_partial(target_ar)
                                batch = self._collate_samples(samples)

            # yield outside the lock and outside watchdog timing
            if exhausted:
                if partial_batches:
                    for b in partial_batches:
                        yield b
                break
            if batch is not None:
                yield batch

    def _pop_batch_for_ar_partial(self, target_ar: str) -> List[Dict]:
        """Like _pop_batch_for_ar but takes ALL remaining samples of that AR
        (for draining at data exhaustion).

        Caller must hold self._cond.
        """
        ar_deque = self._ar_indices[target_ar]
        seqs_to_take = set()
        while ar_deque:
            seqs_to_take.add(ar_deque.popleft())

        samples = []
        new_fifo = deque()
        for seq, ar, sample in self._fifo:
            if seq in seqs_to_take:
                samples.append(sample)
            else:
                new_fifo.append((seq, ar, sample))
        self._fifo = new_fifo
        self._cond.notify_all()
        return samples

    def __len__(self) -> int:
        return len(self.data_loader)

    def close(self) -> None:
        """Stop the prefetch thread and release resources."""
        self._stop_event.set()
        with self._cond:
            self._cond.notify_all()
        if self._prefetch_thread is not None:
            self._prefetch_thread.join(timeout=5.0)
            if self._prefetch_thread.is_alive():
                log.warning(f"[{self.name}] Prefetch thread did not terminate within timeout")
        with self._cond:
            self._fifo.clear()
            self._ar_indices.clear()

        self._check_for_errors()
        self._watchdog.stop()


def get_aspect_ratio_dataloader(
    batch_size: int,
    max_queue_size_per_ar: Optional[int] = None,
    total_max_samples: Optional[int] = None,
    aspect_ratio_dataloader_name: str = "aspect_ratio_dataloader",
    webdataset: bool = True,
    **kwargs,
):
    """Factory for creating aspect ratio aware dataloader.

    Args:
        batch_size (int): Target batch size for yielded batches.
        max_queue_size_per_ar (int, optional): Ignored (kept for config compat).
        total_max_samples (int, optional): Global buffer limit.
            Defaults to batch_size * 50.
        aspect_ratio_dataloader_name (str): Name for logging.
        webdataset (bool): Whether to use webdataset DataLoader.
        **kwargs: Passed to the underlying DataLoader.

    Returns:
        AspectRatioDataLoader: The aspect ratio aware dataloader.
    """
    if webdataset:
        from pid._ext.imaginaire.datasets.webdataset.dataloader import DataLoader as _DataLoader
    else:
        from torch.utils.data import DataLoader as _DataLoader

    if "dataloaders" in kwargs:
        del kwargs["dataloaders"]

    # Underlying dataloader uses batch_size=1; we handle batching.
    kwargs["batch_size"] = 1
    dataloader = _DataLoader(**kwargs)

    return AspectRatioDataLoader(
        data_loader=dataloader,
        batch_size=batch_size,
        total_max_samples=total_max_samples or batch_size * 10,
        name=aspect_ratio_dataloader_name,
    )
