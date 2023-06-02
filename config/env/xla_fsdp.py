from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from xla_strategy.strategy import XlaPjrtFsdpStrategy

def config(cfg: dict):
    callbacks = cfg['trainer']['callbacks'] if 'callbacks' in cfg['trainer'] else []
    
    cfg['trainer'].update({
        'strategy': XlaPjrtFsdpStrategy(),
        'logger': WandbLogger(),
        'callbacks': callbacks.extend([
            LearningRateMonitor(logging_interval='epoch'),
            ModelCheckpoint(monitor='validation/loss') # TODO: 호환됨?
        ])
    })
    return cfg