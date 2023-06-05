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

import os
from typing import Dict, Any, List
import torch
from lightning.pytorch.accelerators.accelerator import Accelerator
from lightning.fabric.utilities.exceptions import MisconfigurationException


class XlaPjrtAccelerator(Accelerator):
    def __init__(self) -> None:
        super().__init__()
        if not XlaPjrtAccelerator.is_available():
            raise MisconfigurationException("XLA accelerator not available")

    def setup_device(self, device: torch.device) -> None:
        pass

    def get_device_stats(self, device: torch.device) -> Dict[str, Any]:
        try:
            import torch_xla.core.xla_model as xm  # type: ignore

            memory_info = xm.get_memory_info(device)
            free_memory = memory_info["kb_free"]
            peak_memory = memory_info["kb_total"] - free_memory
        except:
            free_memory = "unknown"
            peak_memory = "unknown"
        return {
            "avg. free memory (MB)": free_memory,
            "avg. peak memory (MB)": peak_memory,
        }

    def teardown(self) -> None:
        pass

    @staticmethod
    def is_available():
        import torch_xla.core.xla_env_vars as xenv  # type: ignore

        if xenv.PJRT_DEVICE not in os.environ:
            return False
        return XlaPjrtAccelerator.auto_device_count() > 0

    @staticmethod
    def auto_device_count() -> int:
        try:
            import torch_xla.core.xla_env_vars as xenv  # type: ignore

            dev_type = os.environ[xenv.PJRT_DEVICE]
            if dev_type == "CPU":
                dev_count = int(os.environ.get(xenv.CPU_NUM_DEVICES, 1))
            elif dev_type == "GPU":
                dev_count = int(os.environ.get(xenv.GPU_NUM_DEVICES, 1))
            elif dev_type == "TPU":
                dev_count = int(os.environ.get(xenv.TPU_NUM_DEVICES, 8))
            else:
                raise ValueError(f"{xenv.PJRT_DEVICE} is incorrect")
            return dev_count
        except ImportError:
            return 0

    @staticmethod
    def parse_devices(devices: Any) -> int:
        if isinstance(devices, str):
            devices = int(devices)
        if not isinstance(devices, int) or devices <= 0:
            raise TypeError("`devices` should be int >= 0")
        return devices

    @staticmethod
    def get_parallel_devices(dev_indices: int) -> List[torch.device]:
        return [torch.device("xla", index=i) for i in range(dev_indices)]
