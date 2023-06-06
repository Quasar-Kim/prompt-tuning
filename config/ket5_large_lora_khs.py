from model import ket5_large_lora


def config(cfg: dict):
    cfg["model"] = ket5_large_lora
    cfg["runtime_config"]["batch_size"] = 8
    return cfg
