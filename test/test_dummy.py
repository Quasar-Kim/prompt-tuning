import pytest
import torch.distributed as dist
from lightning.pytorch import Trainer, seed_everything
from pytest_cases import fixture, parametrize

import t2tpipe
from t2tpipe.demo.dummy import dummy_model, dummy_task


def _setup(prediction=False):
    return t2tpipe.setup(
        model=dummy_model,
        task=dummy_task,
        runtime_config={
            "batch_size": 16,
            "num_workers": 0,
            "is_gpu": False,
        },
        prediction=prediction,
    )


@pytest.fixture(autouse=True)
def seed():
    seed_everything(42)


@fixture
def cpu_trainer():
    trainer = Trainer(accelerator="cpu", max_epochs=1, log_every_n_steps=1)
    return trainer


@fixture
def gpu_trainer():
    trainer = Trainer(accelerator="gpu", max_epochs=1, log_every_n_steps=1)
    return trainer


@fixture
def cpu_ddp_trainer():
    trainer = Trainer(accelerator="cpu", devices=2, max_epochs=1, log_every_n_steps=1)
    yield trainer
    dist.destroy_process_group()


pytestmark = [
    pytest.mark.filterwarnings("ignore:Your `IterableDataset` has `__len__` defined"),
    pytest.mark.filterwarnings(
        "ignore:.*does not have many workers which may be a bottleneck"
    ),
    pytest.mark.filterwarnings("ignore:GPU available but not used"),
]


@parametrize("trainer", [cpu_trainer, gpu_trainer, cpu_ddp_trainer])
def test_fit(trainer):
    model, dm = _setup()
    trainer.fit(model, dm)


@parametrize("trainer", [cpu_trainer, gpu_trainer, cpu_ddp_trainer])
def test_validate(trainer):
    model, dm = _setup()
    out = trainer.validate(model, dm)
    metrics = out[0]
    assert metrics["validation/average"] == pytest.approx(99.0)


@parametrize("trainer", [cpu_trainer, gpu_trainer, cpu_ddp_trainer])
def test_test(trainer):
    model, dm = _setup()
    out = trainer.test(model, dm)
    metrics = out[0]
    assert metrics["test/average"] == pytest.approx(99.0)


@parametrize("trainer", [cpu_trainer, gpu_trainer, cpu_ddp_trainer])
def test_predict(trainer):
    model, dm = _setup(prediction=True)
    trainer.predict(model, dm)
    assert model.predictions.y_pred == [str(i) for i in range(100)]
