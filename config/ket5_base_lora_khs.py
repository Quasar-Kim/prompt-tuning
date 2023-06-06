from model import ket5_base_lora


def config(cfg: dict):
    cfg["model"] = ket5_base_lora
    cfg["runtime_config"]["batch_size"] = 16
    return cfg
