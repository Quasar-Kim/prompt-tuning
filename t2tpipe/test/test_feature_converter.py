import torch

from t2tpipe.dataclass import (
    EncDecSampleForTrain,
    EncodedSampleForTrain,
    EncodedSampleForPrediction,
    EncDecSampleForPrediction,
)
from t2tpipe.datasource import IterableDataSource
from t2tpipe.feature_converter import EncDecFeatureConverter
from t2tpipe.test.util import dummy_env, dataclass_equal


class TestEncDecFeatureConverter:
    def test_train_output(self):
        bos_token_id = dummy_env.model.tokenizer.bos_token_id
        eos_token_id = dummy_env.model.tokenizer.eos_token_id
        pad_token_id = dummy_env.model.tokenizer.pad_token_id

        inp = EncodedSampleForTrain(
            x=torch.tensor([1, 2, 3]), y=torch.tensor([1, 2, 3])
        )
        source = IterableDataSource([inp])
        source.setup(dummy_env)
        feature_converter = EncDecFeatureConverter()
        feature_converter.setup(dummy_env)
        dp = source.connect(feature_converter)
        out = list(iter(dp))
        assert len(out) == 1
        expected = EncDecSampleForTrain(
            enc_x=torch.tensor([1, 2, 3, eos_token_id]),
            dec_x=torch.tensor([bos_token_id, 1, 2, 3, eos_token_id]),
            y=torch.tensor([1, 2, 3, eos_token_id, pad_token_id]),
        )
        assert dataclass_equal(out[0], expected)

    def test_prediction_output(self, monkeypatch):
        monkeypatch.setattr(dummy_env, "prediction", True)

        bos_token_id = dummy_env.model.tokenizer.bos_token_id
        eos_token_id = dummy_env.model.tokenizer.eos_token_id

        inp = EncodedSampleForPrediction(x=torch.tensor([1, 2, 3]))
        source = IterableDataSource([inp])
        source.setup(dummy_env)
        feature_converter = EncDecFeatureConverter()
        feature_converter.setup(dummy_env)
        dp = source.connect(feature_converter)
        out = list(iter(dp))
        assert len(out) == 1
        expected = EncDecSampleForPrediction(
            enc_x=torch.tensor([1, 2, 3, eos_token_id]),
            dec_x=torch.tensor([bos_token_id]),
        )
        assert dataclass_equal(out[0], expected)
