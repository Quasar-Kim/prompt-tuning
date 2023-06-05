# Copyright The Lightning AI team.
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

import logging
from lightning.fabric.plugins.environments.cluster_environment import ClusterEnvironment
from lightning.fabric.utilities.exceptions import MisconfigurationException
from xla_strategy.accelerator import XlaPjrtAccelerator

log = logging.getLogger(__name__)


class XlaPjrtEnvironment(ClusterEnvironment):
    def __init__(self, *args, **kwargs):
        if not XlaPjrtAccelerator.is_available():
            raise MisconfigurationException("XLA accelerator not available")
        super().__init__(*args, **kwargs)

    @property
    def creates_processes_externally(self):
        return False

    @staticmethod
    def detect() -> bool:
        return XlaPjrtAccelerator.is_available()

    @property
    def main_address(self) -> str:
        raise NotImplementedError

    @property
    def main_port(self) -> int:
        raise NotImplementedError

    def world_size(self) -> int:
        from torch_xla.experimental import pjrt  # type: ignore

        return pjrt.world_size()

    def set_world_size(self, size: int) -> None:
        log.debug("XLA environment does not allow setting world size, ignoring...")

    def global_rank(self) -> int:
        from torch_xla.experimental import pjrt  # type: ignore

        return pjrt.global_ordinal()

    def set_global_rank(self, rank: int) -> None:
        log.debug("XLA environment does not allow setting global rank, ignoring...")

    def local_rank(self) -> int:
        from torch_xla.experimental import pjrt  # type: ignore

        return pjrt.local_ordinal()

    def node_rank(self):
        import torch_xla.core.xla_env_vars as xenv  # type: ignore
        from torch_xla.utils.utils import getenv_as  # type: ignore

        return getenv_as(xenv.HOST_ORDINAL, int, 0)
