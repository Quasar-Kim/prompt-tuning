from time import perf_counter
from lightning.pytorch import LightningDataModule
from rich.console import Console
from rich.table import Table


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
