# type: ignore

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

import contextlib
import io
import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import lightning.pytorch as pl
import torch
from lightning.fabric.plugins import CheckpointIO, XLACheckpointIO
from lightning.fabric.utilities.exceptions import MisconfigurationException
from lightning.fabric.utilities.optimizer import _optimizers_to_device
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.overrides.base import _LightningModuleWrapperBase
from lightning.pytorch.plugins.io.wrapper import _WrappingCheckpointIO
from lightning.pytorch.plugins.precision import PrecisionPlugin
from lightning.pytorch.strategies import DDPStrategy, SingleDeviceStrategy
from lightning.pytorch.strategies.strategy import TBroadcast
from lightning.pytorch.trainer.states import TrainerFn
from lightning.pytorch.utilities.model_helpers import is_overridden
from lightning.pytorch.utilities.parameter_tying import (
    find_shared_parameters,
    set_shared_parameters,
)
from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning_utilities.core.rank_zero import (
    rank_zero_info,
    rank_zero_only,
    rank_zero_warn,
)
from torch import Tensor
from torch.distributed import ReduceOp
from torch.nn import Module

from xla_strategy.accelerator import XlaPjrtAccelerator
from xla_strategy.environment import XlaPjrtEnvironment
from xla_strategy.launcher import XlaPjrtLauncher
from xla_strategy.precision import XlaPrecision
from xla_strategy.util import suppress_stdout

if TYPE_CHECKING and XlaPjrtAccelerator.is_available():
    from torch_xla.distributed.parallel_loader import MpDeviceLoader
else:
    MpDeviceLoader = None

log = logging.getLogger(__name__)


# BUG: 작동 안하고 hang 됨, 디버깅 필요
class XlaPjrtStrategy(DDPStrategy):
    """Strategy for training multiple TPU devices using the :func:`torch_xla.distributed.xla_multiprocessing.spawn`
    method."""

    strategy_name = "xla"

    def __init__(
        self,
        accelerator: "pl.accelerators.Accelerator" = None,
        parallel_devices=None,
        cluster_environment=None,
        checkpoint_io=None,
        precision_plugin: Optional[PrecisionPlugin] = None,
        debug: bool = False,
        sync_module_states: bool = True,
        **_: Any,
    ) -> None:
        if not XlaPjrtAccelerator.is_available():
            raise MisconfigurationException(
                "XLA accelerator not available. Try setting `PJRT_DEVICE`"
            )

        if accelerator is None:
            accelerator = XlaPjrtAccelerator()
        if cluster_environment is None:
            cluster_environment = XlaPjrtEnvironment()
        if checkpoint_io is None:
            checkpoint_io = XLACheckpointIO()
        if precision_plugin is None:
            precision_plugin = XlaPrecision()

        # Parallel devices는 나중에 자동으로 설정됨
        super().__init__(
            accelerator=accelerator,
            parallel_devices=parallel_devices,
            cluster_environment=cluster_environment,
            checkpoint_io=checkpoint_io,
            precision_plugin=precision_plugin,
        )
        self.debug = debug
        self._launched = False
        self._sync_module_states = sync_module_states

    @property
    def checkpoint_io(self) -> CheckpointIO:
        return self._checkpoint_io

    @checkpoint_io.setter
    def checkpoint_io(self, io: Optional[CheckpointIO]) -> None:
        self._checkpoint_io = io

    @property
    def root_device(self) -> torch.device:
        if not self._launched:
            raise RuntimeError(
                "Accessing the XLA device before processes have spawned is not allowed."
            )
        import torch_xla.core.xla_model as xm

        return xm.xla_device()

    def connect(self, model: "pl.LightningModule") -> None:
        # this is called in the spawned process, so no need to use `xmp.MpModelWrapper`
        self.wrapped_model = _LightningModuleWrapperBase(model)
        return super().connect(model)

    def _configure_launcher(self) -> None:
        self._launcher = XlaPjrtLauncher(self)

    def setup(self, trainer: "pl.Trainer") -> None:
        assert self.accelerator
        self.accelerator.setup(trainer)

        if self.debug:
            os.environ["PT_XLA_DEBUG"] = "1"

        assert self.lightning_module
        shared_params = find_shared_parameters(self.lightning_module)
        self.model_to_device()

        set_shared_parameters(self.lightning_module, shared_params)
        self.setup_precision_plugin()

        if self._sync_module_states:
            from torch_xla.experimental import pjrt

            pjrt.broadcast_master_param(self.model)

        if trainer.state.fn == TrainerFn.FITTING:
            self.setup_optimizers(trainer)
            _optimizers_to_device(self.optimizers, self.root_device)

        if self.is_global_zero:
            from torch_xla.experimental import pjrt

            dev_type = pjrt.device_type()
            dev_num = pjrt.global_device_count()
            log.info(
                f"Using XLA PJRT Runtime strategy - type: {dev_type}, count: {dev_num}"
            )

    def _setup_model(self, model: Module) -> Module:  # type: ignore
        return model

    @property
    def distributed_sampler_kwargs(self) -> Dict[str, int]:
        return {"num_replicas": self.world_size, "rank": self.global_rank}

    def process_dataloader(self, dataloader: object) -> "MpDeviceLoader":
        from torch_xla.distributed.parallel_loader import MpDeviceLoader

        if isinstance(dataloader, MpDeviceLoader):
            # dataloader is already wrapped by MpDeviceLoader
            return dataloader

        dataloader = MpDeviceLoader(dataloader, self.root_device)
        # Mimic interface to torch.utils.data.DataLoader
        dataloader.dataset = dataloader._loader.dataset
        dataloader.batch_sampler = getattr(dataloader._loader, "batch_sampler", None)
        return dataloader

    def configure_ddp(self) -> None:
        pass

    def model_to_device(self) -> None:
        self.model = self.wrapped_model.to(self.root_device)

    def barrier(self, name: Optional[str] = None, *args: Any, **kwargs: Any) -> None:
        if not self._launched:
            return

        import torch_xla.core.xla_model as xm

        if name is None:
            # `None` is not supported: "TypeError: _xla_rendezvous(): incompatible function arguments"
            name = ""
        xm.rendezvous(name)

    def broadcast(self, obj: TBroadcast, src: int = 0) -> TBroadcast:
        if not self._launched:
            return obj

        import torch_xla.core.xla_model as xm

        is_tensor = isinstance(obj, Tensor)
        if is_tensor:
            if obj.dim() == 0:
                obj = obj.unsqueeze(0)
            if obj.device.type != "xla":
                obj = obj.to(self.root_device)
        else:
            # support for arbitrary pickle-ables
            buffer = io.BytesIO()
            torch.save(obj, buffer)
            obj = torch.tensor(  # type: ignore[assignment]
                bytearray(buffer.getbuffer()),
                device=self.root_device,
                dtype=torch.float,
            )

        obj = [obj]
        xm.collective_broadcast(obj, root_ordinal=src)
        obj = obj[0]

        if not is_tensor:
            buffer = io.BytesIO(obj.cpu().byte().numpy())
            obj = torch.load(buffer)

        return obj

    def reduce(
        self,
        output: Union[Tensor, Any],
        group: Optional[Any] = None,
        reduce_op: Optional[Union[ReduceOp, str]] = None,
    ) -> Tensor:
        if not isinstance(output, Tensor):
            output = torch.tensor(output, device=self.root_device)

        invalid_reduce_op = (
            isinstance(reduce_op, ReduceOp) and reduce_op != ReduceOp.SUM
        )
        invalid_reduce_op_str = isinstance(
            reduce_op, str
        ) and reduce_op.lower() not in ("sum", "mean", "avg")
        if invalid_reduce_op or invalid_reduce_op_str:
            raise ValueError(
                "Currently, the XLAStrategy only supports `sum`, `mean`, `avg` for the reduce operation, got:"
                f" {reduce_op}"
            )

        import torch_xla.core.xla_model as xm

        output = xm.mesh_reduce("reduce", output, sum)

        if isinstance(reduce_op, str) and reduce_op.lower() in ("avg", "mean"):
            output = output / self.world_size

        return output

    def setup_distributed(self) -> None:
        from torch_xla.experimental.pjrt import using_pjrt

        assert self.parallel_devices is not None
        if using_pjrt() and len(self.parallel_devices) == 1:
            # spawning only 1 device with PjRT is not supported:
            # https://github.com/Lightning-AI/lightning/pull/17408#discussion_r1170671732
            raise NotImplementedError(
                "The `XLAStrategy` does not support running on a single device with the PjRT runtime."
                " Try using all devices or the `SingleDeviceXLAStrategy` strategy"
            )

        self._launched = True
        rank_zero_only.rank = self.global_rank

    def set_world_ranks(self) -> None:
        # accessing global_rank will initialize the XLA computation client. since this is called outside of the spawned
        # processes (by the accelerator connector), we cannot run the code that would normally be here.
        # instead it's done in `setup_distributed`
        pass

    def validation_step(self, *args: Any, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        assert self.model is not None
        with self.precision_plugin.val_step_context():
            return self.model(*args, **kwargs)

    def test_step(self, *args: Any, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        assert self.model is not None
        with self.precision_plugin.test_step_context():
            return self.model(*args, **kwargs)

    def predict_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        assert self.model is not None
        with self.precision_plugin.predict_step_context():
            return self.model(*args, **kwargs)

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        self._pod_progress_bar_force_stdout()

    def remove_checkpoint(self, filepath: _PATH) -> None:
        """Remove checkpoint filepath from the filesystem.

        Args:
            filepath: Path to checkpoint
        """
        if self.local_rank == 0:
            self.checkpoint_io.remove_checkpoint(filepath)

    def all_gather(
        self, tensor: Tensor, group: Optional[Any] = None, sync_grads: bool = False
    ) -> Tensor:
        """Function to gather a tensor from several distributed processes.

        Args:
            tensor: tensor to all-gather.
            group: unused.
            sync_grads: flag that allows users to synchronize gradients for the all-gather operation.
        Return:
            A tensor of shape (world_size, ...)
        """
        if not self._launched:
            return tensor
        if not isinstance(tensor, Tensor):
            raise NotImplementedError(
                f"`{type(self).__name__}.all_gather` is only implemented for tensors. Given {tensor}"
            )
        if tensor.dim() == 0:
            tensor = tensor.unsqueeze(0)
        if tensor.device.type != "xla":
            tensor = tensor.to(self.root_device)

        import torch_xla.core.functions as xf
        import torch_xla.core.xla_model as xm

        return xf.all_gather(tensor) if sync_grads else xm.all_gather(tensor)

    def teardown(self) -> None:
        super().teardown()
        os.environ.pop("PT_XLA_DEBUG", None)

    def _pod_progress_bar_force_stdout(self) -> None:
        # Why is it required? The way `pytorch_xla.distributed` streams logs
        # from different vms to the main worker doesn't work well with tqdm
        # Ref: https://github.com/pytorch/xla/blob/master/torch_xla/distributed/xla_dist.py#L140
        # The print statement seems to force tqdm to flush stdout.
        import torch_xla.core.xla_env_vars as xenv
        from torch_xla.utils.utils import getenv_as

        if self.global_rank == 0 and getenv_as(xenv.TPUVM_MODE, int, 0) == 1:
            print()


class XlaPjrtSingleDeviceStrategy(SingleDeviceStrategy):
    """Strategy for training on a single XLA device."""

    def __init__(
        self,
        device: Any,
        accelerator: Optional["pl.accelerators.Accelerator"] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision_plugin: Optional[PrecisionPlugin] = None,
        debug: bool = False,
    ):
        if not XlaPjrtAccelerator.is_available():
            raise MisconfigurationException("XLA device is not available")
        if isinstance(device, torch.device):
            # unwrap the `torch.device` in favor of `xla_device`
            device = device.index
        import torch_xla.core.xla_model as xm

        if accelerator is None:
            accelerator = XlaPjrtAccelerator()
        if checkpoint_io is None:
            checkpoint_io = XLACheckpointIO()
        if precision_plugin is None:
            precision_plugin = XlaPrecision()

        super().__init__(
            accelerator=accelerator,
            device=xm.xla_device(device),
            checkpoint_io=checkpoint_io,
            precision_plugin=precision_plugin,
        )
        self.debug = debug

    @property
    def checkpoint_io(self) -> CheckpointIO:
        if self._checkpoint_io is None:
            self._checkpoint_io = XLACheckpointIO()
        elif isinstance(self._checkpoint_io, _WrappingCheckpointIO):
            self._checkpoint_io.checkpoint_io = XLACheckpointIO()

        return self._checkpoint_io

    @checkpoint_io.setter
    def checkpoint_io(self, io: Optional[CheckpointIO]) -> None:
        self._checkpoint_io = io

    def setup(self, trainer: "pl.Trainer") -> None:
        assert (
            self.model
        ), "self.model must be set before find_shared_parameters(self.model)"
        shared_params = find_shared_parameters(self.model)
        self.model_to_device()
        set_shared_parameters(self.model, shared_params)
        super().setup(trainer)

        if self.debug:
            os.environ["PT_XLA_DEBUG"] = str(1)

    def teardown(self) -> None:
        super().teardown()
        os.environ.pop("PT_XLA_DEBUG", None)


class XlaPjrtFsdpStrategy(XlaPjrtStrategy):
    """
    XLA + PJRT + FSDP strategy

    Note - Mixed precision:
    원래 하듯이 USE_BFLOAT16 환경 변수 사용시 nan 오류나는 것으로 보임.
    따라서 `compute_precision`을 `__init__`에 넘겨주면 됨.
    (기본값으로 `torch.bfloat16`이 설정되어 있음)
    """

    def __init__(
        self,
        auto_wrap_policy=None,
        flatten_parameters: bool = True,
        compute_precision=torch.bfloat16,
        *args,
        **kwargs,
    ) -> None:
        self._auto_wrap_policy = auto_wrap_policy
        self._flatten_parameters = flatten_parameters
        self._compute_precision = compute_precision
        super().__init__(*args, **kwargs)

    def setup(self, trainer: pl.Trainer) -> None:
        # almost same with FSDP strategy
        assert self.accelerator is not None
        self.accelerator.setup(trainer)

        if trainer.state.fn == TrainerFn.FITTING and self._layer_sync:
            assert self.model is not None
            self.model = self._layer_sync.apply(self.model)

        assert isinstance(self.model, pl.LightningModule)
        self.model = _LightningModuleWrapperBase(self.model)
        if is_overridden("configure_sharded_model", self.lightning_module):
            rank_zero_info(
                "You have overridden `LightningModule.configure_sharded_model` hook. It will assume that all the layers"
                " are already wrapped for sharding and won't wrap the entire model using `FullyShardedDataParallel`."
            )
        else:
            self.model = self._setup_model(self.model)
        self.barrier()

        self.setup_optimizers(trainer)
        _optimizers_to_device(self.optimizers, self.root_device)

        self.setup_precision_plugin()

    def _setup_model(self, model: Module) -> Module:
        from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP

        # TODO: activation checkpointing?
        if self._auto_wrap_policy is not None:
            auto_wrap_policy = self._auto_wrap_policy
            log.debug("using auto wrap policy from __init__")
        elif hasattr(self.lightning_module, "xla_fsdp_auto_wrap_policy"):
            auto_wrap_policy = self.lightning_module.xla_fsdp_auto_wrap_policy
            log.debug("using auto wrap policy specified in model")
        else:
            raise ValueError("auto wrap policy is not specified")
        wrapped_module = FSDP(
            module=model,
            flatten_parameters=self._flatten_parameters,
            mark_step_on_finalization=True,
            auto_wrap_policy=auto_wrap_policy,
            compute_dtype=self._compute_precision,
        )
        if self.is_global_zero:
            log.debug(repr(wrapped_module))
        return wrapped_module

    def model_to_device(self) -> None:
        pass

    def save_checkpoint(
        self, checkpoint: Dict[str, Any], filepath: _PATH, storage_options: Any = None
    ) -> None:
        # TODO: 왜 여기서 barrier 안하면 오류가 발생하는지 알아보기
        self.barrier()
        filepath = Path(filepath)
        assert len(filepath.suffix) > 0

        from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP

        assert isinstance(self.model, FSDP)
        sharded_state_dict = {
            # NOTE: checkpoint['model']은 FSDP wrapper을 제외한 lightning module의 state dict임
            # 따라서 consolidation을 시도하면 오류 발생
            "model": self.model.state_dict(),
            "shard_metadata": self.model.get_shard_metadata(),
        }
        self.checkpoint_io.save_checkpoint(
            sharded_state_dict,
            path=filepath.with_name(f"sharded-state-dict-{self.global_rank}.tmp"),
        )

        self.barrier()

        if self.is_global_zero:
            from torch_xla.distributed.fsdp import consolidate_sharded_model_checkpoints

            with suppress_stdout():
                # print()가 내부에 있는데 무시하기 위해 suppress_stdout 사용
                wrapped_state_dict, _ = consolidate_sharded_model_checkpoints(
                    ckpt_prefix=str(filepath.parent) + "/",
                    ckpt_suffix="sharded-state-dict-*.tmp",
                    save_model=False,
                )
            log.debug("checkpoint consolidated")
            # 임시 파일 제거
            for p in filepath.parent.glob("sharded-state-dict-*.tmp"):
                p.unlink()
            # wrapped_state_dict는 _LightningModuleWrapperBase로 래핑된 LightningModule의 state_dict
            # _LightningModuleWrapperBase는 _forward_module에 LightningModule을 저장함
            # 그리고 실제로 저장될 체크포인트는 아무 wrapper 없는 LightningModule의 state_dict를 저장해야 함
            # 따라서 _forward_module. prefix 제거
            state_dict = OrderedDict()
            for k, v in wrapped_state_dict.items():
                if k.startswith("_forward_module."):
                    new_k = k[len("_forward_module.") :]
                else:
                    new_k = k
                state_dict[new_k] = v
            checkpoint["state_dict"] = state_dict
            self.checkpoint_io.save_checkpoint(
                checkpoint, path=filepath, storage_options=storage_options
            )
            log.debug("saved checkpoint")
        self.barrier()

    @contextlib.contextmanager
    def model_sharded_context(self):
        yield
