def config(cfg: dict):
    cfg["trainer"]["accelerator"] = "gpu"
    cfg["runtime_config"]["is_gpu"] = True
    return cfg
