from model import ket5_small

from config.task.khs import config


def config(cfg: dict):
    cfg = config(cfg)
    cfg["model"] = ket5_small
    cfg["runtime_config"]["batch_size"] = 32
    return cfg
