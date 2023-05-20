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
        metrics.Accuracy(),
        metrics.MacroF1(n_classes=3)
    ]
)

nsmc_with_validation = Task(
    name='nsmc_with_validation',
    source={
        'train': sources.ParquetFileLoader('data/nsmc/train.parquet'),
        'validation': sources.ParquetFileLoader('data/nsmc/validation.parquet'),
        'test': sources.ParquetFileLoader('data/nsmc/test.parquet')
    },
    pipes=[
        pipes.TorchDataPipe(iterpipes.Shuffler),
        pipes.WorkerShardingFilter(),
        pipes.KeyMapper({
            'text': 'x',
            'label': 'y'
        }),
        pipes.YMapper({
            0: '부정',
            1: '긍정'
        })
    ],
    postprocessor=postprocessors.ClassificationPostProcessor({
        '부정': 0,
        '긍정': 1
    }),
    metrics=[
        metrics.Accuracy()
    ]
)

nsmc = Task(
    name='nsmc',
    source={
        'train': sources.ParquetFileLoader('data/nsmc/train_full.parquet'),
        'test': sources.ParquetFileLoader('data/nsmc/test.parquet')
    },
    pipes=[
        pipes.TorchDataPipe(iterpipes.Shuffler),
        pipes.WorkerShardingFilter(),
        pipes.KeyMapper({
            'text': 'x',
            'label': 'y'
        }),
        pipes.YMapper({
            0: '부정',
            1: '긍정'
        })
    ],
    postprocessor=postprocessors.ClassificationPostProcessor({
        '부정': 0,
        '긍정': 1
    }),
    metrics=[
        metrics.Accuracy()
    ]
)