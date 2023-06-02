from model import ket5_base_lora

from config.task.khs import config

def config(cfg: dict):
    cfg = config(cfg)
    cfg['model'] = ket5_base_lora
    cfg['runtime_config']['batch_size'] = 16
    return cfg