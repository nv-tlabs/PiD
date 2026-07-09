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

import io
import re
from typing import Optional

import torch


def pth_decoder(key: str, data: bytes) -> Optional[torch.Tensor]:
    r"""
    Decode a .pth file containing a serialized torch tensor.
    Args:
        key: Data key (filename with extension).
        data: Raw bytes of the .pth file.
    """
    extension = re.sub(r".*[.]", "", key)
    if extension.lower() == "pth":
        return torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    else:
        return None
