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

VIDEO_DATASET_SOURCES = {}

IMAGES_DATASET_SOURCES = {
    "MultiAspect_4K_1M": "data/image_MultiAspect_4K_1M_webdataset/",
    # Add more webdataset-format sources here so you can register them in dataset_definition.py
    # for example:
    # "Rendered_Text": "data/image_Rendered_Text_webdataset/",
    # "Nano_Banana_Image": "data/image_Nano_Banana_Image_webdataset/",
}
