from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger

from xla_strategy.strategy import XlaPjrtFsdpStrategy


def config(cfg: dict):
    callbacks = cfg["trainer"]["callbacks"] if "callbacks" in cfg["trainer"] else []

    cfg["trainer"].update(
        {
            "strategy": XlaPjrtFsdpStrategy(),
            "logger": WandbLogger(),
            "callbacks": callbacks.extend(
                [
                    LearningRateMonitor(logging_interval="epoch"),
                    ModelCheckpoint(monitor="validation/loss"),  # TODO: 호환됨?
                ]
            ),
        }
    )
    cfg["runtime_config"]["is_gpu"] = False
    return cfg
