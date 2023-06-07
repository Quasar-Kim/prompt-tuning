from t2tpipe import Task, datapipe, datasource, metric, postprocessor
from t2tpipe.dataclass import Slot

khs = Task(
    name="khs",
    source={
        "train": datasource.ParquetDataSource("data/khs/train.parquet"),
        "validation": datasource.ParquetDataSource("data/khs/validation.parquet"),
        "prediction": datasource.ParquetDataSource("data/khs/prediction.parquet"),
    },
    pipes=[
        datapipe.Shuffler(),
        datapipe.DistributedShardingFilter(),
        datapipe.WorkerShardingFilter(),
        datapipe.MappingKeyMapper({"text": "x", "label": "y"}),
        datapipe.MappingValueMapper(
            {
                "y": {
                    "hate": "혐오",
                    "offensive": "공격적",
                    "none": "중립",
                }
            }
        ),
        Slot("prefix_adder"),
        datapipe.FeatureTokenizer(),
    ],
    pad_to=128,
    postprocessors=[
        postprocessor.DecoderPostProcessor(),
        Slot("lm_output_slicer"),
        Slot("prefix_remover"),
        postprocessor.ClassificationPostProcessor(
            {
                "혐오": 2,
                "공격적": 1,
                "중립": 0,
            }
        ),
    ],
    metrics=[metric.Accuracy(), metric.F1Macro(n_classes=3)],
)


# NOTE: 이 task의 train set은 전체 train set(train-full.parquet)의 90%로 구성되어 있으며,
# 나머지 10%는 원래는 없는 validation set으로 사용함.
# 따라서 훈련시 성능이 실제보다 낮을 수 있음.
nsmc = Task(
    name="nsmc",
    source={
        "train": datasource.ParquetDataSource("data/nsmc/train.parquet"),
        "validation": datasource.ParquetDataSource("data/nsmc/validation.parquet"),
        "test": datasource.ParquetDataSource("data/nsmc/test.parquet"),
    },
    pipes=[
        datapipe.Shuffler(),
        datapipe.DistributedShardingFilter(),
        datapipe.WorkerShardingFilter(),
        datapipe.MappingKeyMapper({"text": "x", "label": "y"}),
        datapipe.MappingValueMapper(
            {
                "y": {
                    0: "부정",
                    1: "긍정",
                }
            }
        ),
        Slot("prefix_adder"),
        datapipe.FeatureTokenizer(),
    ],
    pad_to=256,
    postprocessors=[
        postprocessor.DecoderPostProcessor(),
        Slot("lm_output_slicer"),
        Slot("prefix_remover"),
        postprocessor.ClassificationPostProcessor(
            {
                "부정": 0,
                "긍정": 1,
            }
        ),
    ],
    metrics=[metric.Accuracy()],
)
