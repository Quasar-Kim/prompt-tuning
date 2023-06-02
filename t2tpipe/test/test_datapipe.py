from dataclasses import asdict
from typing import Union

import torch
import pytest

from t2tpipe.test.util import dummy_env, dataclass_equal
from t2tpipe.datasource import IterableDataSource
from t2tpipe.datapipe import Mapper, MappingKeyMapper, MappingValueMapper, DistributedShardingFilter, FeatureTokenizer, Padder, PadderForEncDecModel
from t2tpipe.tokenizer import Tokenizer
from t2tpipe.dataclass import EncodedSampleForTrain, EncDecSampleForTrain, EncodedSampleForPrediction, EncDecSampleForPrediction

def test_mapper():
    source = IterableDataSource([1, 2, 3])
    source.setup(dummy_env)
    mapper = Mapper(lambda x: x * 2)
    mapper.setup(dummy_env)
    dp = source.connect(mapper)
    assert list(iter(dp)) == [2, 4, 6]

def test_key_mapper():
    source = IterableDataSource([{'x': 1}, {'x': 2}, {'x': 3}])
    source.setup(dummy_env)
    mapper = MappingKeyMapper({
        'x': 'y'
    })
    mapper.setup(dummy_env)
    dp = source.connect(mapper)
    assert list(iter(dp)) == [{'y': 1}, {'y': 2}, {'y': 3}]

class TestValueMapper:
    def test_mapping(self):
        source = IterableDataSource([{'x': 1}, {'x': 2}, {'x': 3}])
        source.setup(dummy_env)
        mapper = MappingValueMapper({
            'x': {
                1: 10,
                2: 20,
                3: 30
            }
        })
        mapper.setup(dummy_env)
        dp = source.connect(mapper)
        assert list(iter(dp)) == [{'x': 10}, {'x': 20}, {'x': 30}]
        
    def test_pass_unknown_key(self):
        source = IterableDataSource([{'x': 1}, {'x': 2}, {'x': 3}])
        source.setup(dummy_env)
        mapper = MappingValueMapper({
            'z': {
                1: 10,
                2: 20,
                3: 30
            }
        })
        mapper.setup(dummy_env)
        dp = source.connect(mapper)
        assert list(iter(dp)) == [{'x': 1}, {'x': 2}, {'x': 3}]

    def test_raise_if_unknown_value_mapping(self):
        source = IterableDataSource([{'x': 1}, {'x': 2}, {'x': 4}])
        source.setup(dummy_env)
        mapper = MappingValueMapper({
            'x': {
                1: 10,
                2: 20,
                3: 30
            }
        })
        mapper.setup(dummy_env)
        dp = source.connect(mapper)
        with pytest.raises(KeyError, match=r'.*Cannot find mapping.*'):
            _ = list(iter(dp))

def test_pipe_len():
    source = IterableDataSource([{'x': 1}, {'x': 2}, {'x': 3}])
    source.setup(dummy_env)
    mapper = MappingValueMapper({
        'x': {
            1: 10,
            2: 20,
            3: 30
        }
    })
    mapper.setup(dummy_env)
    dp = source.connect(mapper)
    assert len(dp) == 3

class TestDistributedShardingFilter:
    def test_sharding(self, monkeypatch):
        def get_world_size():
            return 2
        def get_rank():
            return 0
        monkeypatch.setattr('torch.distributed.is_available', lambda: True)
        monkeypatch.setattr('torch.distributed.is_initialized', lambda: True)
        monkeypatch.setattr('torch.distributed.get_world_size', get_world_size)
        monkeypatch.setattr('torch.distributed.get_rank', get_rank)

        source = IterableDataSource(range(10))
        source.setup(dummy_env)
        filter = DistributedShardingFilter()
        filter.setup(dummy_env)
        dp = source.connect(filter)
        assert list(iter(dp)) == [0, 2, 4, 6, 8]

    def test_drop_uneven(self, monkeypatch):
        monkeypatch.setattr('torch.distributed.is_available', lambda: True)
        monkeypatch.setattr('torch.distributed.is_initialized', lambda: True)
        monkeypatch.setattr('torch.distributed.get_world_size', lambda: 2)
        monkeypatch.setattr('torch.distributed.get_rank', lambda: 0)

        source1 = IterableDataSource(range(5))
        source1.setup(dummy_env)
        filter1 = DistributedShardingFilter()
        filter1.setup(dummy_env)
        dp1 = source1.connect(filter1)
        assert list(iter(dp1)) == [0, 2]

        monkeypatch.setattr('torch.distributed.get_rank', lambda: 1)

        source2 = IterableDataSource(range(5))
        source2.setup(dummy_env)
        filter2 = DistributedShardingFilter()
        filter2.setup(dummy_env)
        dp2 = source2.connect(filter1)
        assert list(iter(dp2)) == [1, 3]

    def test_equal_len(self, monkeypatch):
        monkeypatch.setattr('torch.distributed.is_available', lambda: True)
        monkeypatch.setattr('torch.distributed.is_initialized', lambda: True)
        monkeypatch.setattr('torch.distributed.get_world_size', lambda: 2)
        monkeypatch.setattr('torch.distributed.get_rank', lambda: 0)

        source1 = IterableDataSource(range(5))
        source1.setup(dummy_env)
        filter1 = DistributedShardingFilter()
        filter1.setup(dummy_env)
        dp1 = source1.connect(filter1)
        assert len(dp1) == 4

    def test_prediction_no_sharding(self, monkeypatch):
        monkeypatch.setattr('torch.distributed.is_available', lambda: True)
        monkeypatch.setattr('torch.distributed.is_initialized', lambda: True)
        monkeypatch.setattr('torch.distributed.get_world_size', lambda: 2)
        monkeypatch.setattr('torch.distributed.get_rank', lambda: 0)
        monkeypatch.setattr(dummy_env, 'prediction', True)

        source = IterableDataSource(range(10))
        source.setup(dummy_env)
        filter = DistributedShardingFilter()
        filter.setup(dummy_env)
        dp = source.connect(filter)
        assert list(iter(dp)) == list(range(10))
    
    def test_prediction_len_no_sharding(self, monkeypatch):
        monkeypatch.setattr('torch.distributed.is_available', lambda: True)
        monkeypatch.setattr('torch.distributed.is_initialized', lambda: True)
        monkeypatch.setattr('torch.distributed.get_world_size', lambda: 2)
        monkeypatch.setattr('torch.distributed.get_rank', lambda: 0)
        monkeypatch.setattr(dummy_env, 'prediction', True)

        source = IterableDataSource(range(5))
        source.setup(dummy_env)
        filter = DistributedShardingFilter()
        filter.setup(dummy_env)
        dp = source.connect(filter)
        assert len(dp) == 5

@pytest.fixture
def feature_tokenizer_test_tokenizer():
    class TestTokenizer(Tokenizer):
        def encode(self, text, pad_to=None):
            return [1, 2, 3]

        def encode_batch(self, batch, **kwargs):
            pass

        def decode(self, *args, **kwargs):
            pass

        def decode_batch(self, *args, **kwargs):
            pass
    return TestTokenizer()

class TestFeatureTokenizer:
    def test_train(self, monkeypatch, feature_tokenizer_test_tokenizer):
        monkeypatch.setattr(dummy_env.model, 'tokenizer', feature_tokenizer_test_tokenizer)
        source = IterableDataSource([{'x': 'a', 'y': 'a'}])
        source.setup(dummy_env)
        feature_tokenizer = FeatureTokenizer()
        feature_tokenizer.setup(dummy_env)
        dp = source.connect(feature_tokenizer)
        out = list(iter(dp))
        assert len(out) == 1
        assert isinstance(out[0], EncodedSampleForTrain)
        assert torch.equal(out[0].x, torch.tensor([1, 2, 3]))
        assert torch.equal(out[0].y, torch.tensor([1, 2, 3]))

    def test_prediction(self, monkeypatch, feature_tokenizer_test_tokenizer):
        monkeypatch.setattr(dummy_env.model, 'tokenizer', feature_tokenizer_test_tokenizer)
        monkeypatch.setattr(dummy_env, 'prediction', True)

        source = IterableDataSource([{'x': 'a'}])
        source.setup(dummy_env)
        feature_tokenizer = FeatureTokenizer()
        feature_tokenizer.setup(dummy_env)
        dp = source.connect(feature_tokenizer)
        out = list(iter(dp))
        assert len(out) == 1
        assert isinstance(out[0], EncodedSampleForPrediction)
        assert torch.equal(out[0].x, torch.tensor([1, 2, 3]))

@pytest.fixture
def padder_test_tokenizer():
    class TestTokenizer(Tokenizer):
        @property
        def pad_token(self):
            return '[PAD]'

        @property
        def pad_token_id(self):
            return 0

        def encode(self, *args, **kwargs):
            pass

        def encode_batch(self, *args, **kwargs):
            pass

        def decode(self, *args, **kwargs):
            pass

        def decode_batch(self, *args, **kwargs):
            pass
    return TestTokenizer()

class TestPadder:
    def test_padding(self, monkeypatch, padder_test_tokenizer):
        monkeypatch.setattr(dummy_env.model, 'tokenizer', padder_test_tokenizer)

        sample = EncodedSampleForTrain(
            x=torch.tensor([1, 2, 3]),
            y=torch.tensor([1, 2, 3])
        )
        source = IterableDataSource([sample])
        source.setup(dummy_env)
        padder = Padder(pad_to=5)
        padder.setup(dummy_env)
        dp = source.connect(padder)
        out = list(iter(dp))
        assert isinstance(out[0], EncodedSampleForTrain)
        assert torch.equal(out[0].x, torch.tensor([1, 2, 3, 0, 0]))

    def test_prediction_padding(self, monkeypatch, padder_test_tokenizer):
        monkeypatch.setattr(dummy_env.model, 'tokenizer', padder_test_tokenizer)
        monkeypatch.setattr(dummy_env, 'prediction', True)

        sample = EncodedSampleForPrediction(
            x=torch.tensor([1, 2, 3])
        )
        source = IterableDataSource([sample])
        source.setup(dummy_env)
        padder = Padder(pad_to=5)
        padder.setup(dummy_env)
        dp = source.connect(padder)
        out = list(iter(dp))
        assert isinstance(out[0], EncodedSampleForPrediction)
        assert torch.equal(out[0].x, torch.tensor([1, 2, 3, 0, 0]))

    def test_padder_not_paddable(self, monkeypatch, padder_test_tokenizer):
        monkeypatch.setattr(dummy_env.model, 'tokenizer', padder_test_tokenizer)

        sample = EncodedSampleForTrain(
            x=torch.tensor([1, 2, 3, 4, 5, 6]),
            y=torch.tensor([1, 2, 3])
        )
        source = IterableDataSource([sample])
        source.setup(dummy_env)
        padder = Padder(pad_to=5)
        padder.setup(dummy_env)
        dp = source.connect(padder)
        with pytest.raises(ValueError, match=r'.*pad.*'):
            _ = list(iter(dp))


@pytest.fixture
def tokenizer():
    class TestTokenizer(Tokenizer):
        @property
        def pad_token(self):
            return '[PAD]'

        @property
        def pad_token_id(self):
            return 0

        def encode(self, *args, **kwargs):
            pass

        def encode_batch(self, *args, **kwargs):
            pass

        def decode(self, *args, **kwargs):
            pass

        def decode_batch(self, *args, **kwargs):
            pass
    return TestTokenizer()

class TestPadderForEncDecModel:
    def test_train_output(self, monkeypatch, tokenizer):
        monkeypatch.setattr(dummy_env.model, 'tokenizer', tokenizer)
        monkeypatch.setattr(dummy_env, 'pad_to', 5)

        sample = EncDecSampleForTrain(
            enc_x=torch.tensor([1, 2, 3]),
            dec_x=torch.tensor([1, 2, 3]),
            y=torch.tensor([1, 2, 3])
        )
        source = IterableDataSource([sample])
        source.setup(dummy_env)
        padder = PadderForEncDecModel()
        padder.setup(dummy_env)
        dp = source.connect(padder)
        out = list(iter(dp))
        assert dataclass_equal(out[0], EncDecSampleForTrain(
            enc_x=torch.tensor([1, 2, 3, 0, 0]),
            enc_attention_mask=torch.tensor([1, 1, 1, 0, 0]),
            dec_x=torch.tensor([1, 2, 3, 0, 0]),
            dec_attention_mask=torch.tensor([1, 1, 1, 0, 0]),
            y=torch.tensor([1, 2, 3, 0, 0])
        ))

    def test_prediction_output(self, monkeypatch, tokenizer):
        monkeypatch.setattr(dummy_env.model, 'tokenizer', tokenizer)
        monkeypatch.setattr(dummy_env, 'pad_to', 5)
        monkeypatch.setattr(dummy_env, 'prediction', True)

        sample = EncDecSampleForPrediction(
            enc_x=torch.tensor([1, 2, 3]),
            dec_x=torch.tensor([1])
        )
        source = IterableDataSource([sample])
        source.setup(dummy_env)
        padder = PadderForEncDecModel()
        padder.setup(dummy_env)
        dp = source.connect(padder)
        out = list(iter(dp))
        assert dataclass_equal(out[0], EncDecSampleForPrediction(
            enc_x=torch.tensor([1, 2, 3, 0, 0]),
            enc_attention_mask=torch.tensor([1, 1, 1, 0, 0]),
            dec_x=torch.tensor([1])
        ))

    def test_padder_not_paddable(self, monkeypatch, tokenizer):            
        monkeypatch.setattr(dummy_env.model, 'tokenizer', tokenizer)
        monkeypatch.setattr(dummy_env, 'pad_to', 5)

        sample = EncDecSampleForTrain(
            enc_x=torch.tensor([1, 2, 3, 4, 5, 6]),
            dec_x=torch.tensor([1, 2, 3]),
            y=torch.tensor([1, 2, 3])
        )
        source = IterableDataSource([sample])
        source.setup(dummy_env)
        padder = PadderForEncDecModel()
        padder.setup(dummy_env)
        dp = source.connect(padder)
        with pytest.raises(ValueError, match=r'.*pad.*'):
            _ = list(iter(dp))
