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

import os

# PBSS
import random
import time
from typing import Any, Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from urllib3.exceptions import ProtocolError as URLLib3ProtocolError
from urllib3.exceptions import ReadTimeoutError as URLLib3ReadTimeoutError
from urllib3.exceptions import SSLError as URLLib3SSLError

from pid._ext.imaginaire.utils import log

_OBJECT_STORE_RETRY_EXCEPTIONS = (
    ClientError,
    BotoCoreError,
    URLLib3ReadTimeoutError,
    URLLib3ProtocolError,
    URLLib3SSLError,
    IOError,
    OSError,
)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _retry_sleep(attempt: int) -> float:
    return min(0.2 * 2**attempt + random.uniform(0, 1), 10.0)


def _client_error_code(error: Exception) -> str:
    if isinstance(error, ClientError):
        return str(error.response.get("Error", {}).get("Code", ""))
    return ""


class RetryingStream:
    def __init__(self, client: boto3.client, bucket: str, key: str, retries: Optional[int] = None):  # type: ignore
        r"""Class for loading data in a streaming fashion.
        Args:
            client (boto3.client): Boto3 client
            bucket (str): Bucket where data is stored
            key (str): Key to read
            retries (int): Number of retries
        """
        self.client = client
        self.bucket = bucket
        self.key = key
        self.retries = max(1, retries if retries is not None else _env_int("IMAGINAIRE_S3_STREAM_RETRIES", 3))
        self.name = f"{bucket}/{key}"
        self.content_size = self.get_length()
        self.stream, _ = self.get_stream()
        self._amount_read = 0
        self._stream_read_time = 0.0
        self._read_count = 0
        self._watchdog_min_throughput_mbps = _env_float("RETRYING_STREAM_WATCHDOG_MIN_MBPS", 50.0)
        self._watchdog_min_window_seconds = _env_float("RETRYING_STREAM_WATCHDOG_MIN_SECONDS", 5.0)
        self._watchdog_check_interval = max(1, _env_int("RETRYING_STREAM_WATCHDOG_CHECK_INTERVAL", 50))
        self._watchdog_enabled = os.environ.get("RETRYING_STREAM_WATCHDOG", "1") != "0"
        self._watchdog_reconnect_count = 0
        self._window_start_read_time = 0.0
        self._window_start_bytes = 0

    def _call_with_retries(self, op_name: str, call):
        last_error = None
        for cur_retry_idx in range(self.retries):
            try:
                return call()
            except _OBJECT_STORE_RETRY_EXCEPTIONS as e:
                last_error = e
                code = _client_error_code(e)
                if code in {"404", "NoSuchKey", "NotFound"}:
                    log.warning(f"[{op_name}] missing object {self.name}: {e}", rank0_only=False)
                    raise
                if cur_retry_idx >= self.retries - 1:
                    break
                retry_interval = _retry_sleep(cur_retry_idx)
                log.warning(
                    f"[{op_name}] object-store error for {self.name}: {e} "
                    f"retry: {cur_retry_idx + 1} / {self.retries}, sleeping {retry_interval:.2f}s",
                    rank0_only=False,
                )
                time.sleep(retry_interval)
        log.warning(
            f"[{op_name}] failed for {self.name} after {self.retries} attempts: {last_error}",
            rank0_only=False,
        )
        raise last_error

    def get_length(self) -> int:
        r"""Function for obtaining length of the bytestream"""
        head_obj = self._call_with_retries(
            "head_object",
            lambda: self.client.head_object(Bucket=self.bucket, Key=self.key),
        )
        length = int(head_obj["ContentLength"])
        return length

    def get_stream(self, start_range: int = 0, end_range: Optional[int] = None) -> tuple[Any, int]:
        r"""Function for getting stream in a range
        Args:
            start_range (int): Start index for stream
            end_range (int): End index for stream
        Returns:
            stream (bytes): Stream of data being read
            content_size (int): Length of the bytestream read
        """
        extra_args = {}
        if start_range != 0 or end_range is not None:
            end_range = "" if end_range is None else end_range - 1  # type: ignore
            # Start and end are inclusive in HTTP, convert to Python convention
            range_param = f"bytes={start_range}-{end_range}"
            extra_args["Range"] = range_param
        response = self._call_with_retries(
            "get_object",
            lambda: self.client.get_object(Bucket=self.bucket, Key=self.key, **extra_args),
        )
        content_size = int(response["ContentLength"])
        return response["Body"], content_size

    def _watchdog_reset_stream_if_low_throughput(self, new_position: int) -> None:
        if (
            not self._watchdog_enabled
            or self._read_count % self._watchdog_check_interval != 0
            or self._read_count == 0
            or new_position >= self.content_size
        ):
            return

        window_read_time = self._stream_read_time - self._window_start_read_time
        window_bytes = new_position - self._window_start_bytes
        if window_read_time <= self._watchdog_min_window_seconds or window_bytes <= 0:
            return

        throughput_mbps = (window_bytes / (1024 * 1024)) / window_read_time
        if throughput_mbps >= self._watchdog_min_throughput_mbps:
            return

        self._watchdog_reconnect_count += 1
        log.warning(
            f"[Throughput Watchdog] reconnecting slow stream for {self.name}: "
            f"{throughput_mbps:.1f}MB/s < {self._watchdog_min_throughput_mbps:.1f}MB/s, "
            f"read_time {window_read_time:.1f}s, offset {new_position}/{self.content_size}, "
            f"reconnects={self._watchdog_reconnect_count}",
            rank0_only=False,
        )

        old_stream = self.stream
        try:
            new_stream, _ = self.get_stream(new_position)
            self.stream = new_stream
            if hasattr(old_stream, "close"):
                old_stream.close()
        except _OBJECT_STORE_RETRY_EXCEPTIONS as e:
            log.warning(
                f"[Throughput Watchdog] reconnect failed for {self.name} at offset {new_position}: {e}",
                rank0_only=False,
            )
        self._window_start_read_time = self._stream_read_time
        self._window_start_bytes = new_position

    def read(self, amt: Optional[int] = None) -> bytes:
        r"""Read function for reading the data stream.
        Args:
            amt (int): Amount of data to read
        Returns:
            chunk (bytes): Bytes read
        """

        for cur_retry_idx in range(self.retries):
            read_start = time.monotonic()
            try:
                chunk = self.stream.read(amt)
                self._stream_read_time += time.monotonic() - read_start
                self._read_count += 1
                if amt is not None and amt > 0 and len(chunk) == 0 and self._amount_read != self.content_size:
                    raise IOError("Premature end of stream.")
                new_position = self._amount_read + len(chunk)
                self._watchdog_reset_stream_if_low_throughput(new_position)
                self._amount_read = new_position
                return chunk
            except _OBJECT_STORE_RETRY_EXCEPTIONS as e:
                self._stream_read_time += time.monotonic() - read_start
                if cur_retry_idx >= self.retries - 1:
                    break
                retry_interval = _retry_sleep(cur_retry_idx)
                log.warning(
                    f"[read] stream error for {self.name}: {e} "
                    f"retry: {cur_retry_idx + 1} / {self.retries}, sleeping {retry_interval:.2f}s",
                    rank0_only=False,
                )
                time.sleep(retry_interval)
                if hasattr(self.stream, "close"):
                    self.stream.close()
                self.stream, _ = self.get_stream(self._amount_read)

        log.warning(
            f"[read] failed after {self.retries} attempts at byte offset "
            f"{self._amount_read} / {self.content_size} for {self.name}",
            rank0_only=False,
        )
        raise IOError(f"Unable to read {self.name} after {self.retries} attempts.")
