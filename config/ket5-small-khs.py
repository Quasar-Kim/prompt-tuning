def config(cfg: dict):
    cfg['task'] = 'task.khs'
    cfg['model'] = 'model.ket5_small'
    cfg['runtime_config'] = {
        'batch_size': 32,
        'num_workers': 4,
        'num_warmup_epochs': 5,
        'lr': 1e-3
    }
    cfg['trainer']['max_epochs'] = 50
    return cfg