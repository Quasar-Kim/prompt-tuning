from model import dummy_model
from task import khs

def config(cfg: dict):
    cfg['task'] = khs
    cfg['model'] = dummy_model
    cfg['runtime_config'].update({
        'batch_size': 32,
        'num_workers': 4
    })
    cfg['trainer'].update({
        'max_epochs': 1
    })
    return cfg