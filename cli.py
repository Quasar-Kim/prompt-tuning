'''
Usage: python cli.py [stage] --seed [seed] --config [config module1] --config [config module2] ...
Example: python cli.py fit --config config.model.ket5_small --config config.data.khs

Required keys:
    - task: task dataclass
    - model: model dataclass
    - trainer: trainer configuration, dict
    - runtime_config: runtime configuration, dict

Example config file:
    def config(cfg):
        cfg['task'] = task_registry.get('khs')
        return cfg
'''

from typing import List, Optional
import argparse
import importlib
import json
from pathlib import Path

from prettyprinter import cpprint
from lightning.pytorch import Trainer, seed_everything

import t2tpipe
from t2tpipe.dataclass import Task, Model

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', type=str, help='stage to run')
    parser.add_argument('--config', type=str, action='append', help='config scripts')
    parser.add_argument('--override', type=str)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    return args

def parse_config(config_modules: List[str], override: Optional[str] = None):
    cfg = {
        'task': None,
        'model': None,
        'trainer': {},
        'runtime_config': {}
    }
    cfg = run_config_scripts(cfg, config_modules)
    if override is not None:
        cfg = json.loads(override)
    validate_cfg(cfg)
    return cfg

def run_config_scripts(cfg, config_modules: List[str]):
    for config_module in config_modules:
        module = importlib.import_module(config_module)
        cfg = module.config(cfg)
    return cfg

def validate_cfg(cfg: dict):
    assert 'model' in cfg and isinstance(cfg['model'], Model)
    assert 'task' in cfg and isinstance(cfg['task'], Task)
    assert 'trainer' in cfg and isinstance(cfg['trainer'], dict)
    assert 'runtime_config' in cfg and isinstance(cfg['runtime_config'], dict)

def report_config(parsed_cfg: dict, log_dir: Optional[str]):
    task_name = parsed_cfg['task'].name
    model_name = parsed_cfg['model'].name
    config_for_reporting = {
        **parsed_cfg,
        'task': f'<Task name=\'{task_name}\'>',
        'model': f'<Model name=\'{model_name}\'>',
    }
    print('parsed config:')
    cpprint(config_for_reporting)

    if log_dir is None:
        log_path = Path.cwd() / 'config.json'
    else:
        log_path = Path(log_dir) / 'config.json'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('w') as f:
        json.dump(config_for_reporting, f, indent=4)

def run_stage(stage, parsed_cfg: dict):
    model, dm = t2tpipe.setup(
        model=parsed_cfg['model'],
        task=parsed_cfg['task'],
        runtime_config=parsed_cfg['runtime_config']
    )
    trainer = Trainer(**parsed_cfg['trainer'])
    report_config(parsed_cfg, log_dir=trainer.log_dir)
    stage = getattr(trainer, stage)
    stage(model, dm)

if __name__ == '__main__':
    args = parse_arguments()
    parsed_cfg = parse_config(args.config, args.override)
    seed_everything(args.seed)
    run_stage(args.stage, parsed_cfg)
