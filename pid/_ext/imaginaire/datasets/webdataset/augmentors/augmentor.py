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

import time
from collections.abc import Iterable
from typing import Any, Generator, Optional

from pid._ext.imaginaire.utils import log


class Augmentor:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        original_call = getattr(cls, "__call__", None)
        if original_call is None or getattr(original_call, "_has_call_timer", False):
            return

        def timed_call(self, *args: Any, **kwds: Any) -> Any:
            if self._should_log_call_time():
                start = time.perf_counter()
                try:
                    return original_call(self, *args, **kwds)
                finally:
                    duration_ms = (time.perf_counter() - start) * 1000.0
                    if duration_ms > 1000:
                        log.info(f"{self.__class__.__name__}.__call__ took {duration_ms:.2f} ms!")
            return original_call(self, *args, **kwds)

        timed_call._has_call_timer = True  # type: ignore[attr-defined]
        cls.__call__ = timed_call

    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        r"""Base augmentor class

        Args:
            input_keys (list): List of input keys
            output_keys (list): List of output keys
            args (dict): Arguments associated with the augmentation. If args['log_call_time'] = True, it will print the call time.
        """
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.args = args
        self._log_call_time = self._extract_log_flag(args)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise ValueError("Augmentor not implemented")

    @staticmethod
    def _extract_log_flag(args: Optional[dict]) -> bool:
        default_value = False
        if isinstance(args, dict):
            return bool(args.get("log_call_time", default_value))
        try:
            return bool(args.get("log_call_time", default_value))  # type: ignore[call-arg]
        except Exception:
            return default_value

    def _should_log_call_time(self) -> bool:
        return self._log_call_time


class IterableAugmentor:
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        r"""Base augmentor class

        Args:
            input_keys (list): List of input keys
            output_keys (list): List of output keys
            args (dict): Arguments associated with the augmentation
        """
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.args = args
        self.is_generator = True

    def __call__(self, data: Iterable) -> Generator:
        r"""Example usage:

        for data_dict in data:
            # Do something to data_dict
            data_dict["input"] = data_dict["raw_sequence"][:, :-1]
            data_dict["target"] = data_dict["raw_sequence"][:, 1:]
            # Skip sample if needed
            if data_dict["input"].shape[1] < 64:
                continue
            # Construct a generator
            yield data_dict
        """
        raise ValueError("Augmentor not implemented")
