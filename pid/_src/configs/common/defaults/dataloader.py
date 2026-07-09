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


from hydra.core.config_store import ConfigStore

from pid._ext.imaginaire.dataloaders.cached_replay_dataloader import get_cached_replay_dataloader
from pid._ext.imaginaire.lazy_config import LazyCall as L
from pid._src.datasets.data_sources.mock_data import get_image_dataset

_IMAGE_LOADER = L(get_cached_replay_dataloader)(
    dataset=L(get_image_dataset)(
        resolution="1024",
    ),
    batch_size=2,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    webdataset=False,
    cache_replay_name="image_dataloader",
)


MOCK_DATA_IMAGE_ONLY_CONFIG = _IMAGE_LOADER


def register_training_and_val_data():
    cs = ConfigStore()
    cs.store(group="data_train", package="dataloader_train", name="mock_image", node=MOCK_DATA_IMAGE_ONLY_CONFIG)
    cs.store(group="data_val", package="dataloader_val", name="mock_image", node=MOCK_DATA_IMAGE_ONLY_CONFIG)
