'''
Usage: python cli.py [stage] --config [config module] --config [config module] ...
Example: python cli.py fit --config config.model.ket5-small --config config.data.khs

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

from lightning.pytorch import Trainer

import t2tpipe
from t2tpipe.dataclass import Task, Model

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', type=str, help='stage to run')
    parser.add_argument('--config', type=str, action='append', help='config scripts')
    parser.add_argument('--override', type=str)

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

def run_stage(stage, parsed_cfg: dict):
    model, dm = t2tpipe.setup(
        model=parsed_cfg['model'],
        task=parsed_cfg['task'],
        runtime_config=parsed_cfg['runtime_config']
    )
    trainer = Trainer(**parsed_cfg['trainer'])
    stage = getattr(trainer, stage)
    stage(model, dm)

if __name__ == '__main__':
    args = parse_arguments()
    parsed_cfg = parse_config(args.config, args.override)
    run_stage(args.stage, parsed_cfg)
