from model import ket5_large_lora

from config.task.khs import config


def config(cfg: dict):
    cfg = config(cfg)
    cfg["model"] = ket5_large_lora
    cfg["runtime_config"]["batch_size"] = 8
    return cfg
