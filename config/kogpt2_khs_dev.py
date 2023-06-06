from model import kogpt2


def config(cfg: dict):
    # OVERRIDES
    cfg["trainer"]["max_epochs"] = 1

    cfg["model"] = kogpt2
    cfg["runtime_config"]["batch_size"] = 2
    return cfg
