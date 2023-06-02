from t2tpipe.demo.dummy import dummy_model, dummy_task

def config(cfg: dict):
    cfg['task'] = dummy_task
    cfg['model'] = dummy_model
    cfg['runtime_config'].update({
        'batch_size': 128,
        'num_workers': 1
    })
    cfg['trainer'].update({
        'max_epochs': 1
    })
    return cfg