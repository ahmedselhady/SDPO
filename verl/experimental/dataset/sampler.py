# Copyright 2025 Amazon.com Inc and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import random
import logging
from abc import abstractmethod
from collections.abc import Sized

from omegaconf import DictConfig
from torch.utils.data import Sampler

from verl import DataProto

logger = logging.getLogger(__name__)


class AbstractSampler(Sampler[int]):
    """Abstract interface for custom samplers."""

    @abstractmethod
    def __init__(
        self,
        data_source: Sized,
        data_config: DictConfig,
    ):
        pass


class AbstractCurriculumSampler(AbstractSampler):
    """Experimental interface for curriculum learning samplers."""

    @abstractmethod
    def update(self, batch: DataProto) -> None:
        pass


class ParallelGroupBatchSampler(Sampler[list[int]]):
    """Batch dataset rows without splitting groups of parallel examples.

    Groups are packed into full batches. Any groups that cannot be packed into a
    full batch are left out for that epoch, matching the trainer's drop-last
    behavior while ensuring that no parallel group crosses a batch boundary.
    """

    def __init__(self, data_source: Sized, data_config: DictConfig):
        self.data_source = data_source
        self.batch_size = int(data_config.get("gen_batch_size") or data_config.train_batch_size)
        self.group_key = data_config.parallel_group_batching.get("group_key", "parallel_id")
        self.seed = data_config.get("seed") or 0
        self.epoch = 0

        dataframe = getattr(data_source, "dataframe", None)
        if dataframe is None:
            raise ValueError(
                "parallel_group_batching requires a dataset exposing its source columns as `dataframe`; "
                "set data.parallel_group_batching.enabled=False or provide a compatible dataset"
            )
        columns = dataframe.column_names if hasattr(dataframe, "column_names") else dataframe.columns
        if self.group_key not in columns:
            raise ValueError(
                f"parallel_group_batching.group_key={self.group_key!r} is not a dataset column"
            )
        group_values = dataframe[self.group_key]
        groups = {}
        for index, group_id in enumerate(group_values):
            if group_id is None or str(group_id).strip().lower() in {"", "nan", "none", "<na>"}:
                group_id = ("__single_row__", index)
            groups.setdefault(group_id, []).append(index)
        self.groups = list(groups.values())

        oversized = [len(group) for group in self.groups if len(group) > self.batch_size]
        if oversized:
            raise ValueError(
                f"parallel group size {max(oversized)} exceeds training batch size {self.batch_size}; "
                "increase data.train_batch_size or reduce the parallel group size"
            )
        if len(self) == 0:
            raise ValueError(
                f"No complete parallel-group batch fits batch size {self.batch_size}; "
                "check parallel_id values and data.train_batch_size"
            )
        packed_rows = sum(len(batch) for batch in self._make_batches(self.epoch))
        if packed_rows < len(data_source):
            logger.warning(
                "Parallel-group batching packed %s of %s rows into full batches; %s rows will be skipped this epoch. "
                "Adjust data.train_batch_size or parallel group sizes to reduce skipped data.",
                packed_rows,
                len(data_source),
                len(data_source) - packed_rows,
            )

    def _make_batches(self, epoch: int) -> list[list[int]]:
        rng = random.Random(None if self.seed is None else int(self.seed) + epoch)
        groups = [group[:] for group in self.groups]
        groups.sort(key=len, reverse=True)

        batches: list[list[int]] = []
        batch_sizes: list[int] = []
        for group in groups:
            for batch_index, current_size in enumerate(batch_sizes):
                if current_size + len(group) <= self.batch_size:
                    batches[batch_index].extend(group)
                    batch_sizes[batch_index] += len(group)
                    break
            else:
                batches.append(group[:])
                batch_sizes.append(len(group))

        full_batches = [batch for batch, size in zip(batches, batch_sizes) if size == self.batch_size]
        rng.shuffle(full_batches)
        for batch in full_batches:
            rng.shuffle(batch)
        return full_batches

    def __iter__(self):
        batches = self._make_batches(self.epoch)
        yield from batches
        self.epoch += 1

    def __len__(self) -> int:
        return len(self._make_batches(self.epoch))

    def state_dict(self) -> dict:
        return {"epoch": self.epoch}

    def load_state_dict(self, state_dict: dict) -> None:
        self.epoch = int(state_dict.get("epoch", 0))
