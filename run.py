import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import RichModelSummary, ModelCheckpoint, LearningRateMonitor
from data.core import create_experiment
import models
import tasks


if __name__ == '__main__':
    pl.seed_everything(42)
    lit_model, dm = create_experiment(
        task=tasks.khs,
        model=models.keT5Small,
        pad_to=128,
        batch_size=2,
        is_gpu=False,
        hparams={
            'lr': 1e-4,
            'num_warmup_epochs': 10
        }
    )
    trainer = pl.Trainer(
        accelerator='cpu',
        max_epochs=10,
        callbacks=[
            RichModelSummary(),
            LearningRateMonitor(logging_interval='epoch')
        ],
        # logger=WandbLogger(project='khs-ket5')
    )
    trainer.fit(lit_model, dm)