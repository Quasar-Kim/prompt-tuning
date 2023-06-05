import torch

from t2tpipe.postprocessor import ClassificationPostProcessor, LMOutputSlicer
from t2tpipe.dataclass import ModelTrainOutput, ModelPredictionOutput
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
    def test_output(self):
        postprocessor = LMOutputSlicer()
        postprocessor.setup(dummy_env)
        y = ModelTrainOutput(
            x=torch.tensor([[1], [2]]),
            y=torch.tensor([[100], [200]]),
            y_pred=torch.tensor([[1, 100], [2, 300]]),
            loss=torch.tensor(0.0),
        )
        out = postprocessor(y)
        assert out.y_pred.tolist() == [[100], [300]]
