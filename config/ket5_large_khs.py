from model import ket5_large


def config(cfg: dict):
    cfg["model"] = ket5_large
    cfg["runtime_config"]["batch_size"] = 8
    return cfg
