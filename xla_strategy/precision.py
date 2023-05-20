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

from typing import Any, Literal
import torch
import os
from torch import Tensor
from lightning.fabric.plugins.precision.precision import Precision
from lightning.fabric.utilities.types import Optimizable
from lightning.fabric.plugins.precision.utils import _convert_fp_tensor
from lightning.fabric.utilities.exceptions import MisconfigurationException
from lightning_utilities.core.apply_func import apply_to_collection

from xla_strategy.accelerator import XlaPjrtAccelerator


class XlaPrecision(Precision):
    """Precision plugin with XLA."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if not XlaPjrtAccelerator.is_available():
            raise MisconfigurationException('XLA accelerator not available')
        super().__init__(*args, **kwargs)

    def optimizer_step(
        self,
        optimizer: Optimizable,
        use_xm: bool = True,
        **kwargs: Any,
    ) -> Any:
        import torch_xla.core.xla_model as xm
        if use_xm:
            return xm.optimizer_step(optimizer, optimizer_args=kwargs)
        else:
            return optimizer.step(**kwargs)
    
class XlaBf16Precision(XlaPrecision):
    """Plugin that enables mixed bf16 with XLA."""

    precision: Literal["bf16-mixed"] = "bf16-mixed"

    def __init__(self) -> None:
        super().__init__()
        os.environ["XLA_USE_BF16"] = "1"

    def convert_input(self, data: Any) -> Any:
        return apply_to_collection(data, function=_convert_fp_tensor, dtype=Tensor, dst_type=torch.bfloat16)

    def convert_output(self, data: Any) -> Any:
        return apply_to_collection(data, function=_convert_fp_tensor, dtype=Tensor, dst_type=torch.get_default_dtype())
    
    def teardown(self) -> None:
        os.environ.pop("XLA_USE_BF16", None)