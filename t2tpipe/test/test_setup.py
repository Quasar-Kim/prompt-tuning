import pytest

from t2tpipe.setup import setup as t2tpipe_setup
from t2tpipe.demo.dummy import dummy_model, dummy_task
from t2tpipe.dataclass import Slot
from t2tpipe.datapipe import Shuffler

demo_runtime_config = {
    "num_warmup_epochs": 1,
    "lr": 1,
    "batch_size": 1,
    "num_workers": 0,
    "is_gpu": False,
}


class Testslot:
    def test_remove_unused_slot(self, monkeypatch):
        monkeypatch.setattr(dummy_task, "pipes", [Slot("hello")])
        model, dm = t2tpipe_setup(
            model=dummy_model, task=dummy_task, runtime_config=demo_runtime_config
        )

        assert dm.env.task.pipes == []

    def test_raise_on_unused_required_slot(self, monkeypatch):
        monkeypatch.setattr(dummy_task, "pipes", [Slot("hello", required=True)])

        with pytest.raises(ValueError, match=r".*slot.*"):
            model, dm = t2tpipe_setup(
                model=dummy_model, task=dummy_task, runtime_config=demo_runtime_config
            )

    def test_slot_replace(self, monkeypatch):
        shuffler1 = Shuffler()
        shuffler2 = Shuffler()
        monkeypatch.setattr(dummy_task, "pipes", [Slot("hello"), shuffler2])
        monkeypatch.setattr(dummy_model, "pipes", {"hello": shuffler1})
        model, dm = t2tpipe_setup(
            model=dummy_model, task=dummy_task, runtime_config=demo_runtime_config
        )
        assert dm.env.task.pipes == [shuffler1, shuffler2]
