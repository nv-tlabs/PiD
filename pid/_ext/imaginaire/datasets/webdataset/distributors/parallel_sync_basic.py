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
import random
import time

import torch

try:
    from megatron.core import parallel_state

    USE_MEGATRON = True
except ImportError:
    USE_MEGATRON = False

from webdataset.utils import pytorch_worker_info

from pid._ext.imaginaire.datasets.webdataset.distributors.basic import ShardlistBasic
from pid._ext.imaginaire.datasets.webdataset.utils.misc import repeat_list
from pid._ext.imaginaire.utils import log


class ShardlistBasicParallelSync(ShardlistBasic):
    r"""
    An iterable dataset that parses and yields tar files.
    This distributor is based on ShardlistBasic.
    Additionally, it allows users to synchronize inputs for context/tensor parallelism.
    This is achieved by specifying the context/tensor parallel group size during initialization.

    Ranks of the same pp/tp/cp group will have the same dp rank and thus share the same group id,
    ensuring they process the same data samples.
    """

    def __init__(self, **kwargs):
        r"""Create a basic ShardList with parallel sync support.

        Args:
            shuffle (bool): shuffle samples before iterating.
            split_by_node (bool): split shards by node if True
            split_by_worker (bool): split shards by worker if True
            resume_flag (bool): If enabled, resumes from a specific iteration and epoch number.
            verbose (bool): Prints some logs if true
            is_infinite_loader (bool): If true, creates an infinite dataloader.
            max_epochs (int): Infinite dataloader is created with max_epochs number of epochs.
            repeat_url (bool): If true, each worker will receive the same number of batches by repeating urls.
        """
        super().__init__(**kwargs)
        self.enable_parallel()

    def enable_parallel(self):
        """Enable parallel synchronization for context/tensor parallelism.

        Ranks of the same pp/tp/cp group will have the same dp rank and thus share the same group id.
        The size of the group is how many GPUs we use to process one batch of data.
        """
        self.group_id = parallel_state.get_data_parallel_rank()
        self.group_size = torch.distributed.get_world_size() // parallel_state.get_data_parallel_world_size()

    def obtain_url_list(self):
        r"""Return an iterator over the shards with parallel sync support."""

        rank, world_size, worker_id, num_workers = pytorch_worker_info()

        # Calculate the number of groups based on group size
        num_groups = world_size // self.group_size

        # Setting epoch and start index
        if self.resume_flag:
            self.epoch = int(os.environ.get("WDS_EPOCH_NUM", 0))
            # This tells us number of chunks that have been seen by one GPU
            self.start_index = int(os.environ.get("WDS_START_INDEX", 0)) // self.chunk_size

        urls = self.urls
        num_urls = len(urls)

        # nworkers_all is no longer world_size * num_workers, since self.group_size workers duplicate
        nworkers_all = num_groups * num_workers

        if self.verbose:
            log.info(f"Total {nworkers_all} effective workers (num_groups={num_groups}, num_workers={num_workers})")
            log.info(f"Rank {rank}, group_id {self.group_id}, group_size {self.group_size}, worker {worker_id}")

        if self.repeat_url:
            # Extending urls so that each worker receives the same number of batches.
            # This serves the job of ddp_equalize.
            num_urls_per_process = (num_urls + nworkers_all - 1) // nworkers_all
            extended_url_list_size = num_urls_per_process * nworkers_all
            urls = repeat_list(urls, extended_url_list_size)

        # Splits the urls by group id and worker id.
        # This ensures workers in the same group see the same urls.
        if self.split_by_node:
            urls = urls[self.group_id :: num_groups]
        if self.split_by_worker:
            # avoid worker in iterable dataset retrieving the same url
            urls = urls[worker_id::num_workers]

        if self.verbose:
            log.info("List of urls (before shuffle)")
            log.info(urls[0:2])

        if self.shuffle:
            # Shuffle based on the group id to ensure workers in the same group see the same shuffle
            random.Random(self.group_id * num_workers + worker_id).shuffle(urls)

        # This tells us the number of chunks seen by one worker.
        # Do not iterate over the seen chunks.
        start_index_per_worker = self.start_index // num_workers
        if not self.is_infinite_loader:
            urls = urls[start_index_per_worker:]

        if self.verbose:
            log.info("List of urls (after shuffle)")
            log.info(urls[0:2])
            log.info(
                f"Rank {rank}, group {self.group_id}, worker {worker_id} of {num_workers}, "
                f"group_size {self.group_size} got {len(urls)} urls"
            )

        return urls

    def __iter__(self):
        url_list = self.obtain_url_list()

        if self.is_infinite_loader:
            for _ in range(self.max_epochs):
                cur_time = int(time.time())
                random.Random(cur_time).shuffle(url_list)
                for url in url_list:
                    yield dict(url=url)
        else:
            for url in url_list:
                yield dict(url=url)
