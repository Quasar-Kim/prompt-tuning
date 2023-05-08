from functools import partial
from typing import Union
import torch
from torchdata.datapipes import iter as iterpipes
from local_types import *

class EncDecFeatureConverter:
    def __init__(self, pad_to: Union[int, None]):
        self.pad_to = pad_to

    def _convert_feature(self, sample: FeatureConverterInput, tokenizer: Tokenizer):
        assert tokenizer.bos_token_id is not None
        assert tokenizer.pad_token_id is not None
        return {
            'enc_x': self._pad(sample['x'], tokenizer.pad_token_id),
            'enc_attention_mask': self._pad(sample['x_attention_mask'], 0),
            'dec_x': self._pad([tokenizer.bos_token_id] + sample['y'], tokenizer.pad_token_id),
            'dec_attention_mask': self._pad([1] + sample['y_attention_mask'], 0),
            'y': self._pad(sample['y'] + [tokenizer.pad_token_id], tokenizer.pad_token_id)
        }
    
    def _pad(self, l: 'list[int]', pad_value: int):
        t = torch.tensor(l)
        if self.pad_to is None:
            return t
        pad_len = self.pad_to - t.shape[-1]
        return torch.nn.functional.pad(t, (0, pad_len), value=pad_value)

    def __call__(self, dp, dm):
        return iterpipes.Mapper(dp, partial(self._convert_feature, tokenizer=dm.tokenizer))