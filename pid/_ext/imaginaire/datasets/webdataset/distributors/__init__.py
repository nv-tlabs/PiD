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

from pid._ext.imaginaire.datasets.webdataset.distributors.basic import ShardlistBasic
from pid._ext.imaginaire.datasets.webdataset.distributors.multi_aspect_ratio import (
    ShardlistMultiAspectRatio,
)
from pid._ext.imaginaire.datasets.webdataset.distributors.multi_aspect_ratio_v2 import (
    ShardlistMultiAspectRatioInfinite,
)
from pid._ext.imaginaire.datasets.webdataset.distributors.parallel_sync_basic import ShardlistBasicParallelSync
from pid._ext.imaginaire.datasets.webdataset.distributors.parallel_sync_multi_aspect_ratio import (
    ShardlistMultiAspectRatioParallelSync,
)

distributors_list = {
    "basic": ShardlistBasic,
    "multi_aspect_ratio": ShardlistMultiAspectRatio,
    "multi_aspect_ratio_infinite": ShardlistMultiAspectRatioInfinite,
    "parallel_sync_basic": ShardlistBasicParallelSync,
    "parallel_sync_multi_aspect_ratio": ShardlistMultiAspectRatioParallelSync,
}
