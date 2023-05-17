from data import sources, pipes, postprocessors, metrics
from torchdata.datapipes import iter as iterpipes
from local_types import *

khs = Task(
    name='khs',
    source={
        'train': sources.ParquetFileLoader('data/khs/train.parquet'),
        'validation': sources.ParquetFileLoader('data/khs/validation.parquet')
    },
    pipes=[
        pipes.TorchDataPipe(iterpipes.Shuffler),
        pipes.WorkerShardingFilter(),
        pipes.KeyMapper({
            'text': 'x',
            'label': 'y'
        }),
        pipes.YMapper({
            'hate': '혐오',
            'offensive': '공격적',
            'none': '중립'
        })
    ],
    postprocessor=postprocessors.ClassificationPostProcessor({
        '혐오': 2,
        '공격적': 1,
        '중립': 0
    }),
    metrics=[
        metrics.Accuracy()
    ]
)