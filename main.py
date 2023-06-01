import sys
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI
from lightning.pytorch.demos.boring_classes import DemoModel, BoringDataModule

import tasks
import models
from data.core import create_experiment

class MyCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        parser.add_argument('--experiment', type=dict)
        parser.add_argument('--ckpt_path', enable_path=True, default=None)

def cli_main():
    cli = MyCLI(
        DemoModel, 
        BoringDataModule, 
        save_config_kwargs={'overwrite': True}, 
        run=False,
        args=sys.argv[2:]
    )
    
    subcommand = sys.argv[1]
    fn = getattr(cli.trainer, subcommand)

    cfg = cli.config['experiment']
    print(cfg)
    task = getattr(tasks, cfg.pop('task'))
    model = getattr(models, cfg.pop('model'))
    lit_model, dm = create_experiment(
        task,
        model,
        **cfg
    )
    fn(lit_model, dm, ckpt_path=cli.config['ckpt_path'])
    
if __name__ == "__main__":
    cli_main()