from local_types import *

class ClassificationPostProcessor:
    def __init__(self, mapping: 'dict[str, int]', invalid_class_label=100):
        self.mapping = mapping
        self.invalid_class_label = invalid_class_label
    
    def __call__(self, y_or_y_pred: str, tokenizer: Tokenizer) -> int:
        try:
            return self.mapping[y_or_y_pred]
        except KeyError:
            return self.invalid_class_label