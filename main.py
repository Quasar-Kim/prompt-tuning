from lightning.pytorch.callbacks.progress.tqdm_progress import Tqdm
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.demos.boring_classes import DemoModel, BoringDataModule
import tasks
import models
from data.core import create_experiment

class MyCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        parser.add_argument('--experiment', type=dict)
        parser.add_argument('--ckpt_path', enable_path=True, default=None)

class StatOnlyProgressBar(TQDMProgressBar):
    def _init_tqdm(self, bar):
        bar.total = None
        return bar

    def init_train_tqdm(self) -> Tqdm:
        bar = super().init_train_tqdm()
        bar = self._init_tqdm(bar)
        return bar
    
    def init_validation_tqdm(self) -> Tqdm:
        bar = super().init_validation_tqdm()
        bar = self._init_tqdm(bar)
        return bar

def cli_main():
    cli = MyCLI(
        DemoModel, 
        BoringDataModule, 
        save_config_kwargs={'overwrite': True}, 
        run=False
    )
    cfg = cli.config['experiment']
    task = getattr(tasks, cfg.pop('task'))
    model = getattr(models, cfg.pop('model'))
    lit_model, dm = create_experiment(
        task,
        model,
        **cfg
    )
    cli.trainer.fit(lit_model, dm, ckpt_path=cli.config['ckpt_path'])
    
if __name__ == "__main__":
    cli_main()