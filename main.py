from lightning.pytorch.cli import LightningArgumentParser, LightningCLI
from lightning.pytorch.demos.boring_classes import DemoModel, BoringDataModule
import tasks
import models
from data.core import create_experiment

class MyCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        parser.add_argument('--experiment')


def cli_main():
    cli = MyCLI(DemoModel, BoringDataModule, run=False)
    cfg = cli.config['experiment']
    task = getattr(tasks, cfg.pop('task'))
    model = getattr(models, cfg.pop('model'))
    lit_model, dm = create_experiment(
        task,
        model,
        **cfg
    )
    cli.trainer.fit(lit_model, dm)
    
if __name__ == "__main__":
    cli_main()