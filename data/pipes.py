from functools import partial
from torchdata.datapipes import iter as iterpipes
from torch.utils.data.datapipes.iter.sharding import SHARDING_PRIORITIES
from local_types import *
    
class TorchDataPipe:
    def __init__(self, cls, *args, **kwargs):
        self.cls = cls
        self.args = args
        self.kwargs = kwargs

    def __call__(self, dp, dm):
        return self.cls(dp, *self.args, **self.kwargs)
    
class WorkerShardingFilter:
    def __call__(self, dp, dm):
        return iterpipes.ShardingFilter(dp, sharding_group_filter=SHARDING_PRIORITIES.MULTIPROCESSING)
    
class KeyMapper:
    def __init__(self, mapping: 'dict[str, str]'):
        self.mapping = mapping

    def _map(self, sample):
        mapped = sample
        for k, v in self.mapping.items():
            mapped[v] = mapped[k]
            del mapped[k]
        return mapped

    def __call__(self, dp, dm):
        return iterpipes.Mapper(dp, self._map)

class YMapper:
    def __init__(self, mapping: 'dict[str, str]'):
        self.mapping = mapping

    def _map(self, sample):
        sample['y'] = self.mapping[sample['y']]
        return sample

    def __call__(self, dp, dm):
        return iterpipes.Mapper(dp, self._map)
    
class FeatureTokenizer:  
    def __call__(self, dp, dm):
        return iterpipes.Mapper(dp, partial(self._tokenize, tokenizer=dm.tokenizer))
    
    def _tokenize(self, sample, tokenizer):
        if 'y' in sample:
            return self._tokenize_train_sample(sample, tokenizer)
        else:
            return self._convert_inference_sample(sample, tokenizer)
        
    def _tokenize_train_sample(self, sample, tokenizer: Tokenizer):
        tokenized_sample = sample.copy()
        for k, v in sample.items():
            if not isinstance(v, str):
                continue
            tokenized = tokenizer.encode(v)
            tokenized_sample[k] = tokenized.input_ids
            tokenized_sample[f'{k}_attention_mask'] = tokenized.attention_mask
        return tokenized_sample
        
    def _convert_inference_sample(self, sample, tokenizer):
        raise NotImplementedError()