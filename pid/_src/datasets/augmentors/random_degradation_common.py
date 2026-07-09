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

import random
from typing import Optional

import numpy as np

from pid._ext.imaginaire.datasets.webdataset.augmentors.augmentor import Augmentor


def _build_degradations(degradations):
    """Build degradations from LazyCall objects or dict configs.

    Supports both:
    1. LazyCall objects (new unified format)
    2. Dict configs with 'type' and 'params' (legacy format for backward compatibility)
    """
    # Import here to avoid circular import
    from omegaconf import DictConfig

    from pid._ext.imaginaire.lazy_config import LazyCall
    from pid._ext.imaginaire.lazy_config.instantiate import instantiate

    # Convert to list to handle OmegaConf objects
    degradations = list(degradations) if not isinstance(degradations, list) else degradations

    for i, degradation in enumerate(degradations):
        # Check if it's a list/tuple (nested degradation group)
        if isinstance(degradation, (list, tuple)) or (
            hasattr(degradation, "__iter__")
            and not hasattr(degradation, "items")
            and not isinstance(degradation, LazyCall)
        ):
            # Recursively build degradations in the sublist
            degradations[i] = _build_degradations(degradation)
        elif isinstance(degradation, DictConfig):  # LazyDict
            # New unified format: LazyCall object, instantiate it directly
            degradations[i] = instantiate(degradation)
        else:
            raise ValueError(f"Unknown degradation type: {type(degradation)}")

    return degradations


class DegradationsWithShuffle(Augmentor):
    """Apply random degradations to input, with degradations being shuffled.

    Degradation groups are supported. The order of degradations within the same
    group is preserved. For example, if we have degradations = [a, b, [c, d]]
    and shuffle_idx = None, then the possible orders are

    ::

        [a, b, [c, d]]
        [a, [c, d], b]
        [b, a, [c, d]]
        [b, [c, d], a]
        [[c, d], a, b]
        [[c, d], b, a]

    Modified keys are the attributed specified in "keys".

    Args:
        input_keys (list): A list specifying the keys whose values are modified.
        output_keys (list): List of output keys (not used, same as input_keys).
        args (dict): Should contain:
            - degradations (list[dict]): The list of degradations.
            - shuffle_idx (list | None): The degradations corresponding to
                these indices are shuffled. If None, all degradations are shuffled.
    """

    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

        self.keys = input_keys
        degradations = args.get("degradations", []) if args else []
        shuffle_idx = args.get("shuffle_idx", None) if args else None

        self.degradations = _build_degradations(degradations)

        if shuffle_idx is None:
            self.shuffle_idx = list(range(0, len(degradations)))
        else:
            self.shuffle_idx = shuffle_idx

    def __call__(self, data_dict):
        """Call this augmentor."""
        # shuffle degradations
        if len(self.shuffle_idx) > 0:
            shuffle_list = [self.degradations[i] for i in self.shuffle_idx]
            np.random.shuffle(shuffle_list)
            for i, idx in enumerate(self.shuffle_idx):
                self.degradations[idx] = shuffle_list[i]

        # apply degradations to input
        for degradation in self.degradations:
            if isinstance(degradation, (tuple, list)):
                for subdegrdation in degradation:
                    data_dict = subdegrdation(data_dict)
            else:
                data_dict = degradation(data_dict)

        return data_dict

    def __repr__(self):
        """Print the basic information of the augmentor."""
        repr_str = self.__class__.__name__
        repr_str += f"(degradations={self.degradations}, keys={self.keys}, shuffle_idx={self.shuffle_idx})"
        return repr_str


class DegradationsRandomChoice(Augmentor):
    """Apply random degradations to input, with degradations being chosen randomly.

    Degradation groups are supported. The order of degradations within the same
    group is preserved. For example, if we have degradations = [a, b, c]
    then we randomly choose one of them to apply.

    Args:
        input_keys (list): A list specifying the keys whose values are modified.
        output_keys (list): List of output keys (not used, same as input_keys).
        args (dict): Should contain:
            - degradations (list[dict]): The list of degradations.
    """

    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

        self.keys = input_keys
        degradations = args.get("degradations", []) if args else []
        self.degradations = _build_degradations(degradations)

    def __call__(self, data_dict):
        """Call this augmentor."""
        # randomly choose one of the degradations to apply
        degradation = random.choice(self.degradations)

        # apply degradations to input
        data_dict = degradation(data_dict)

        return data_dict

    def __repr__(self):
        """Print the basic information of the augmentor."""
        repr_str = self.__class__.__name__
        repr_str += f"(degradations={self.degradations}, keys={self.keys})"
        return repr_str
