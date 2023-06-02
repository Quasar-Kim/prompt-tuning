from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

def config(cfg: dict):
    callbacks = cfg['trainer']['callbacks'] if 'callbacks' in cfg['trainer'] else []
    
    cfg['trainer'].update({
        'accelerator': 'gpu',
        'logger': WandbLogger(),
        'callbacks': callbacks.extend([
            LearningRateMonitor(logging_interval='epoch'),
            ModelCheckpoint(monitor='validation/loss')
        ])
    })
    return cfg