from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

def config(cfg: dict):
    callbacks = cfg['trainer']['callbacks'] if 'callbacks' in cfg['trainer'] else []
    
    cfg['trainer'].update({
        'accelerator': 'tpu',
        'devices': 8,
        'precision': 'bf16-mixed',
        'logger': WandbLogger(),
        'callbacks': callbacks.extend([
            LearningRateMonitor(logging_interval='epoch'),
            ModelCheckpoint(monitor='validation/loss')
        ])
    })
    return cfg