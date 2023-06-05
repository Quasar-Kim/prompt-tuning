import pytest

import torch
from lightning.pytorch import Trainer, seed_everything

import t2tpipe
from t2tpipe.demo.dummy import dummy_model, dummy_task


@pytest.fixture
def trainer():
    seed_everything(42)
    trainer = Trainer(accelerator="cpu", max_epochs=1, log_every_n_steps=1)
    return trainer


@pytest.fixture
def train_model_and_dm():
    return t2tpipe.setup(
        model=dummy_model,
        task=dummy_task,
        runtime_config={
            "batch_size": 64,
            "num_workers": 0,
            "is_gpu": False,
        },
    )


@pytest.fixture
def prediction_model_and_dm():
    return t2tpipe.setup(
        model=dummy_model,
        task=dummy_task,
        runtime_config={
            "batch_size": 64,
            "num_workers": 0,
            "is_gpu": False,
        },
        prediction=True,
    )


pytestmark = [
    pytest.mark.filterwarnings("ignore:Your `IterableDataset` has `__len__` defined"),
    pytest.mark.filterwarnings(
        "ignore:.*does not have many workers which may be a bottleneck"
    ),
]


def test_fit(trainer, train_model_and_dm):
    model, dm = train_model_and_dm
    trainer.fit(model, dm)


def test_validate(trainer, train_model_and_dm):
    model, dm = train_model_and_dm
    out = trainer.validate(model, dm)
    metrics = out[0]
    assert metrics["validation/average"] == pytest.approx(99.0)


def test_test(trainer, train_model_and_dm):
    model, dm = train_model_and_dm
    out = trainer.test(model, dm)
    metrics = out[0]
    assert metrics["test/average"] == pytest.approx(99.0)


def test_predict(trainer, prediction_model_and_dm):
    model, dm = prediction_model_and_dm
    trainer.predict(model, dm)
    assert model.predictions.y_pred == [str(i) for i in range(100)]
