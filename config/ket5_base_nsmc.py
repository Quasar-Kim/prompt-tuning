from model import ket5_base


def config(cfg: dict):
    cfg["model"] = ket5_base
    cfg["runtime_config"]["batch_size"] = 16
    return cfg
