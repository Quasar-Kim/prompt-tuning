from model import ket5_small


def config(cfg: dict):
    cfg["model"] = ket5_small
    cfg["runtime_config"]["batch_size"] = 16
    return cfg
