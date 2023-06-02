from t2tpipe import datasource, datapipe, postprocessor, metric, Task

khs = Task(
    name='khs',
    source={
        'train': datasource.ParquetDataSource('data/khs/train.parquet'),
        'validation': datasource.ParquetDataSource('data/khs/validation.parquet')
    },
    pipes=[
        datapipe.Shuffler(),
        datapipe.DistributedShardingFilter(),
        datapipe.WorkerShardingFilter(),
        datapipe.MappingKeyMapper({
            'text': 'x',
            'label': 'y'
        }),
        datapipe.MappingValueMapper({
            'y': {
                'hate': '혐오',
                'offensive': '공격적',
                'none': '중립'
            }
        }),
        datapipe.FeatureTokenizer()
    ],
    pad_to=128,
    postprocessor=postprocessor.ClassificationPostProcessor({
        '혐오': 2,
        '공격적': 1,
        '중립': 0
    }),
    metrics=[
        metric.Accuracy(),
        metric.F1Macro(n_classes=3)
    ]
)
