from local_types import *
import torch

class EncDecFeatureConverter:
    tokenizer: Tokenizer
    pad_to: int = None

    def convert(self, sample: TextInferenceSample | TextTrainSample):
        assert self.tokenizer is not None
        if 'y' in sample:
            return self._convert_train_sample(sample)
        else:
            return self._convert_inference_sample(sample)
        
    def _convert_train_sample(self, sample: TextTrainSample) -> EncDecLabeledSample:
        x = self.tokenizer.encode(sample['x'])
        y = self.tokenizer.encode(sample['y'])
        converted = {
            'enc_x': self._to_tensor_with_padding(x['input_ids']),
            'enc_attention_mask': self._to_tensor_with_padding(x['attention_mask']),
            'dec_x': self._to_tensor_with_padding([self.tokenizer.bos_token_id] + y['input_ids']),
            'dec_attention_mask': self._to_tensor_with_padding([1] + y['attention_mask']),
            'y': self._to_tensor_with_padding(y['input_ids'] + [self.tokenizer.pad_token_id]),
        }
        
        for k, v in sample.items():
            if k not in ['x', 'y']:
                converted[k] = torch.tensor(v)

        return converted
    
    def _to_tensor_with_padding(self, l: list):
        if self.pad_to is not None:
            l = self._pad(l)
        return torch.tensor(l)
    
    def _pad(self, l: list[int]):
        # NOTE: [0] * n에서 n <= 0이면 빈 list를 만듦
        return l + [self.tokenizer.pad_token_id] * (self.pad_to - len(l))
        
    def _convert_inference_sample(self):
        raise NotImplementedError()

