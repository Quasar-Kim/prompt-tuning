import os
from transformers import T5Tokenizer

class KeT5Tokenizer:
    def __init__(self, parallelism=False):
        if not parallelism:
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        self._tokenizer = T5Tokenizer.from_pretrained('KETI-AIR/ke-t5-small')

    def encode_train_sample(self, x, y, pad=True, pad_to=None):
        opts = dict()
        if pad:
            opts['padding'] = 'max_length'
            if pad_to is not None:
                opts['max_length'] = pad_to
        inp_enc_x = self._tokenizer(x, **opts)
        inp_dec_x = self._tokenizer('<s>' + y, **opts)
        inp_y = self._tokenizer(y, **opts)
        return {
            'input_ids': inp_enc_x['input_ids'],
            'attention_mask': inp_enc_x['attention_mask'],
            'decoder_input_ids': inp_dec_x['input_ids'],
            'decoder_attention_mask': inp_dec_x['attention_mask'],
            'y': inp_y['input_ids']
        }
    
    def decode(self, *args, **kwargs):
        return self._tokenizer.decode(*args, **kwargs)
    
    def encode(self, *args, **kwargs):
        return self._tokenizer.encode(*args, **kwargs)
    
    def __call__(self, *args, **kwargs):
        return self._tokenizer(*args, **kwargs)