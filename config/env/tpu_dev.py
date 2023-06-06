def config(cfg: dict):
    cfg["trainer"].update(
        {"accelerator": "tpu", "devices": 8, "precision": "bf16-mixed"}
    )
    cfg["runtime_config"]["is_gpu"] = False
    return cfg
