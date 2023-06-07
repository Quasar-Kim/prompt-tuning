from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger


def config(cfg: dict):
    callbacks = cfg["trainer"]["callbacks"] if "callbacks" in cfg["trainer"] else []

    cfg["trainer"].update(
        {
            "accelerator": "gpu",
            "logger": WandbLogger(),
            "callbacks": callbacks.extend(
                [
                    LearningRateMonitor(logging_interval="epoch"),
                    ModelCheckpoint(monitor="validation/loss"),
                ]
            ),
        }
    )
    cfg["runtime_config"]["is_gpu"] = True
    return cfg
