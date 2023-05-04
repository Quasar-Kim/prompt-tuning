import torch
from local_types import *

class ClassificationPostProcessor:
    tokenizer: Tokenizer

    def __init__(self, mapping: dict[str, int], invalid_class_label=100):
        self.mapping = mapping
        self.invalid_class_label = invalid_class_label
        self.tokenizer = None

    def __call__(self, outputs, batch: EncDecLabeledSample):
        assert self.tokenizer is not None
        logits = outputs['logits']
        preds = torch.argmax(logits, dim=-1) # (B, N)
        preds = self._y_to_class_indices(preds)
        y = self._y_to_class_indices(batch['y'])
        return {**outputs, 'preds': preds, 'y': y}
    
    def _y_to_class_indices(self, y):
        str_preds = self.tokenizer.decode_batch(y, remove_special_tokens=True)
        indices = []
        for p in str_preds:
            try:
                indices.append(self.mapping[p])
            except KeyError:
                indices.append(self.invalid_class_label)
        return torch.tensor(indices)
        
            
