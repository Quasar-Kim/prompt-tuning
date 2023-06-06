from task import nsmc


def config(cfg: dict):
    cfg["task"] = nsmc
    cfg["runtime_config"] = {
        **cfg["runtime_config"],
        "num_workers": 4,
        "num_warmup_epochs": 5,
        "lr": 1e-3,
    }
    cfg["trainer"]["max_epochs"] = 25
    return cfg
