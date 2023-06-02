def config(cfg: dict):
    cfg['trainer'].update({
        'accelerator': 'cpu',
        'devices': 2
    })
    cfg['runtime_config']['is_gpu'] = False
    return cfg