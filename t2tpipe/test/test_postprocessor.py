import torch

from t2tpipe.dataclass import ModelPredictionOutput, ModelTrainOutput
from t2tpipe.postprocessor import ClassificationPostProcessor, LMOutputSlicer
from t2tpipe.test.util import dummy_env


class TestClassificationPostProcessor:
    def test_train_output(self):
        postprocessor = ClassificationPostProcessor(
            {
                "y": {
                    "부정": 0,
                    "긍정": 1,
                },
                "y_pred": {
                    "부정": 3,
                    "긍정": 4,
                },
            }
        )
        postprocessor.setup(dummy_env)
        y = ModelTrainOutput(y=["부정"], y_pred=["긍정"], loss=torch.tensor(0.0), x=[])
        out = postprocessor(y)
        assert out.y == [0]
        assert out.y_pred == [4]

    def test_prediction_output(self, monkeypatch):
        monkeypatch.setattr(dummy_env, "prediction", True)
        postprocessor = ClassificationPostProcessor(
            {
                "x": {
                    "부정": 0,
                    "긍정": 1,
                },
                "y_pred": {
                    "부정": 3,
                    "긍정": 4,
                },
            }
        )
        postprocessor.setup(dummy_env)
        y = ModelPredictionOutput(x=["부정"], y_pred=["긍정"])
        out = postprocessor(y)
        assert out.x == [0]
        assert out.y_pred == [4]

    def test_unk(self):
        postprocessor = ClassificationPostProcessor(
            {
                "y_pred": {
                    "부정": 0,
                    "긍정": 1,
                }
            },
            unk_id=-100,
        )
        postprocessor.setup(dummy_env)
        y = ModelTrainOutput(y=["중립"], y_pred=["긍정"], loss=torch.tensor(0.0), x=[])
        out = postprocessor(y)
        assert out.y == [-100]


class TestLMOutputSlicer:
    def test_train_output(self):
        postprocessor = LMOutputSlicer(output_prefix="예측: ")
        postprocessor.setup(dummy_env)
        y = ModelTrainOutput(
            x=["문장: 안녕? 예측: 인사"],
            y=["문장: 안녕? 예측: 인사"],
            y_pred=["문장: 안녕? 예측: 안부"],
            loss=torch.tensor([0.0]),
        )
        out = postprocessor(y)
        assert out.y == ["인사"]
        assert out.y_pred == ["안부"]

    def test_prediction_output(self, monkeypatch):
        monkeypatch.setattr(dummy_env, "prediction", True)
        postprocessor = LMOutputSlicer(output_prefix="예측: ")
        postprocessor.setup(dummy_env)
        y = ModelPredictionOutput(
            x=["문장: 안녕? 예측:"],
            y_pred=["문장: 안녕? 예측: 안부"],
        )
        out = postprocessor(y)
        assert out.y_pred == ["안부"]
