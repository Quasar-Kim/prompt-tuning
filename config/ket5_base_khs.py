from model import ket5_base

from config.task.khs import config

def config(cfg: dict):
    cfg = config(cfg)
    cfg['model'] = ket5_base
    cfg['runtime_config']['batch_size'] = 16
    return cfg