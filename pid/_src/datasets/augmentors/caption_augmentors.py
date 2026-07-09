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

# Augmentors for loading captions from webdataset for PixelDiT T2I training.
#
# Caption data in webdataset shards is stored as JSON files with structure:
#   {"prompt": "A long 200-300 word caption...", "file_name": "xxx.jpg",
#    "prompt_medium": "A 50-200 word caption...",    (optional)
#    "prompt_short": "A <50 word caption."}          (optional)
#
# CaptionExtractor randomly samples among available prompt lengths to increase
# text diversity during training. Sampling probabilities are configurable.
#
# Usage in augmentor_provider.py:
#   pipeline["caption_extractor"] = L(CaptionExtractor)(
#       input_keys=["caption"],
#       args={"output_key": "caption"},
#   )

import random

from pid._ext.imaginaire.datasets.webdataset.augmentors.augmentor import Augmentor
from pid._ext.imaginaire.utils import log


class CaptionExtractor(Augmentor):
    """Extract prompt text from caption JSON dict with multi-length sampling.

    Webdataset loads caption/*.json files as dicts with one or more prompt fields:
      - "prompt": long caption (200-300 words, always present)
      - "prompt_medium": medium caption (50-200 words, optional)
      - "prompt_short": short caption (<50 words, optional)

    When multiple lengths are available, one is sampled per-sample according to
    configurable probabilities. This encourages the model to handle diverse caption
    lengths during training.

    args:
        output_key: key to store the extracted prompt (default: "caption")
        prompt_fields: ordered list of (field_name, weight) pairs for sampling.
            Default: [("prompt", 0.5), ("prompt_medium", 0.3), ("prompt_short", 0.2)]
            Fields missing from the JSON are skipped and their weight redistributed.
    """

    # Default sampling weights: equal probability for each available length
    _DEFAULT_PROMPT_FIELDS = [
        ("prompt", 1.0),
        ("prompt_medium", 1.0),
        ("prompt_short", 1.0),
    ]

    def __call__(self, data_dict: dict) -> dict | None:
        output_key = (self.args or {}).get("output_key", "caption")
        prompt_fields = self._DEFAULT_PROMPT_FIELDS

        if self.input_keys:
            present_input_keys = [input_key for input_key in self.input_keys if input_key in data_dict]
            if len(present_input_keys) != 1:
                log.warning(
                    "Skipping sample because CaptionExtractor expected exactly one "
                    f"caption input key from {self.input_keys}, got {present_input_keys}. "
                    f"__key__={data_dict.get('__key__')}, __url__={data_dict.get('__url__')}"
                )
                return None

            # retrieve the key that is present in the data_dict
            caption_data = data_dict[present_input_keys[0]]
        else:
            caption_data = data_dict.get("caption")

        if caption_data is None:
            log.warning("no caption data found")
            return None

        # Handle caption from s3, which is nested in a dict
        def is_nested(caption_data):
            return (
                isinstance(caption_data, dict)
                and ("caption" in caption_data and isinstance(caption_data["caption"], dict))
                or ("captions" in caption_data and isinstance(caption_data["captions"], dict))
            )

        while is_nested(caption_data):
            if "caption" in caption_data:
                caption_data = caption_data["caption"]
            else:
                caption_data = caption_data["captions"]

        # Handle raw string (no sampling needed)
        if isinstance(caption_data, str):
            data_dict[output_key] = caption_data
            return data_dict

        if not isinstance(caption_data, dict):
            log.warning("caption data is not a dict")
            return None

        # Collect available fields and their weights
        available = []
        for field, weight in prompt_fields:
            text = caption_data.get(field)
            if text:
                available.append((text, weight))

        if not available:
            log.warning(f"no valid prompt found, caption data is {caption_data}")
            return None

        # Weighted random sampling among available lengths
        texts, weights = zip(*available)
        (prompt,) = random.choices(texts, weights=weights, k=1)

        data_dict[output_key] = prompt
        return data_dict
