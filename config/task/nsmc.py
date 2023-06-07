from task import nsmc


def config(cfg: dict):
    cfg["task"] = nsmc
    cfg["runtime_config"].update(
        {
            "num_workers": 1,
            "num_warmup_epochs": 5,
            "lr": 1e-3,
        }
    )
    cfg["trainer"]["max_epochs"] = 25
    return cfg
