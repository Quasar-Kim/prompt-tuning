from task import khs

def config(cfg: dict):
    cfg['task'] = khs
    cfg['runtime_config'] = {
        'num_workers': 4,
        'num_warmup_epochs': 5,
        'lr': 1e-3
    }
    cfg['trainer']['max_epochs'] = 50
    return cfg