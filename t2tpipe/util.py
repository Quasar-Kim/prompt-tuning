import dataclasses
from time import perf_counter
from typing import Dict, List, Protocol, TypeVar

import torch
from lightning.pytorch import LightningDataModule
from rich.console import Console
from rich.table import Table
from torch.utils.data import default_collate


def benchmark_datamodule(datamodule: LightningDataModule):
    datamodule.setup(stage="fit")
    loader = datamodule.train_dataloader()
    # warmup
    print("warming up...")
    _ = list(iter(loader))
    # benchmark
    print("benchmarking...")
    start = perf_counter()
    samples = list(iter(loader))
    end = perf_counter()
    print("done!")
    duration = end - start
    n_batches = len(samples)
    if len(loader) != n_batches:
        print(
            f"WARNING: number of actual batches ({len(samples)}) != len(loader) ({len(loader)})"
        )

    table = Table(title="Train DataLoader Benchmark Results")
    table.add_column("Metric", style="magenta")
    table.add_column("Value", style="green")
    table.add_row("Number of Batches", str(n_batches))
    table.add_row("Duration", f"{duration:.2f} s")
    table.add_row("Duration per Batch", f"{duration / n_batches:.2f} s")
    table.add_row("Batch per Second", f"{n_batches / duration:.2f} batches/s")

    console = Console()
    console.print(table)


class DataClassLike(Protocol):
    __dict__: dict


_DATACLASS = TypeVar("_DATACLASS", bound=DataClassLike)


class TensorDataClassLike(Protocol):
    __dict__: Dict[str, torch.Tensor]


_TENSOR_DATACLASS = TypeVar("_TENSOR_DATACLASS", bound=TensorDataClassLike)


def collate_dataclass(samples: List[_DATACLASS]) -> _DATACLASS:
    assert dataclasses.is_dataclass(samples[0])
    dict_samples = [s.__dict__ for s in samples]
    collated = default_collate(dict_samples)
    return dataclasses.replace(samples[0], **collated)


def join_tensor_dataclass(samples: List[_TENSOR_DATACLASS]) -> _TENSOR_DATACLASS:
    assert dataclasses.is_dataclass(samples[0])
    dict_samples = [s.__dict__ for s in samples]
    concatenated = {}
    for k in dict_samples[0].keys():
        values = []
        for s in dict_samples:
            values.append(s[k])
        concatenated[k] = join_tensors(values)
    return dataclasses.replace(samples[0], **concatenated)


def dataclass_to_cpu(dataclass: _DATACLASS) -> _DATACLASS:
    assert dataclasses.is_dataclass(dataclass)
    entries = {}
    for k, v in dataclass.__dict__.items():
        assert isinstance(v, torch.Tensor)
        entries[k] = v.cpu()
    return dataclasses.replace(dataclass, **entries)


def join_tensors(tensors: List[torch.Tensor], dim: int = 0) -> torch.Tensor:
    assert isinstance(tensors[0], torch.Tensor)
    if tensors[0].ndim > 0:
        return torch.cat(tensors, dim=dim)
    return torch.stack(tensors, dim=dim)
