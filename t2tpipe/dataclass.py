from typing import Any, List, Dict, Mapping, Optional, TYPE_CHECKING
from dataclasses import dataclass
from torch import Tensor

if TYPE_CHECKING:
    from t2tpipe.base import BaseLightningModule
    from t2tpipe.datamodule import T2tPipeDataModule
    from t2tpipe.datapipe import TransformDataPipe
    from t2tpipe.datasource import DataSource
    from t2tpipe.feature_converter import FeatureConverter
    from t2tpipe.metric import Metric
    from t2tpipe.postprocessor import PostProcessor
    from t2tpipe.tokenizer import Tokenizer
else:
    BaseLightningModule = None
    T2tPipeDataModule = None
    Tokenizer = None
    FeatureConverter = None
    DataSource = None
    TransformDataPipe = None
    PostProcessor = None
    Metric = None


@dataclass
class EncodedSampleForPrediction:
    x: Tensor


@dataclass
class EncodedSampleForTrain:
    x: Tensor
    y: Tensor


@dataclass
class EncDecSampleForPrediction:
    enc_x: Tensor
    dec_x: Tensor
    enc_attention_mask: Optional[Tensor] = None


@dataclass
class EncDecSampleForTrain:
    enc_x: Tensor
    dec_x: Tensor
    y: Tensor
    enc_attention_mask: Optional[Tensor] = None
    dec_attention_mask: Optional[Tensor] = None


@dataclass
class ModelPredictionOutput:
    x: Tensor
    pred: Tensor


@dataclass
class ModelTrainOutput:
    y: Tensor
    y_pred: Tensor
    loss: Tensor


@dataclass
class BatchTextPrediction:
    x: List[str]
    pred: List[str]


# @dataclass
# class TextTrainOutput:
#     y: List[str]
#     y_pred: List[str]


@dataclass
class PostProcessedOutput:
    y: Tensor
    y_pred: Tensor


@dataclass
class Task:
    name: str
    source: Mapping[str, DataSource]
    pipes: List[TransformDataPipe]
    pad_to: Optional[int] = None
    postprocessor: Optional[PostProcessor] = None
    metrics: Optional[List[Metric]] = None


@dataclass
class Model:
    name: str
    feature_converter: FeatureConverter
    module: BaseLightningModule
    tokenizer: Tokenizer
    padder: Optional[TransformDataPipe] = None
    pipes: Optional[Dict[str, TransformDataPipe]] = None


@dataclass
class Env:
    model: Model
    task: Task
    datamodule: T2tPipeDataModule
    runtime_config: Dict[str, Any]
    prediction: bool
    pad_to: Optional[int] = None


@dataclass
class Slot:
    name: str
    required: bool = False


# @dataclass
# class TokenizerEncoding:
#     input_ids: List[int]
#     attention_mask: List[int]

# @dataclass
# class BatchTokenizerEncoding:
#     input_ids: List[List[int]]
#     attention_mask: List[List[int]]
