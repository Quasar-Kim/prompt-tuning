def config(cfg: dict):
    cfg['trainer']['accelerator'] = 'cpu'
    cfg['runtime_config']['is_gpu'] = False
    return cfg