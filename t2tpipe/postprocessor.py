from abc import ABC, abstractmethod
from typing import Any, Dict

from t2tpipe.mixin import SetupMixin

class PostProcessor(ABC, SetupMixin):
    @abstractmethod
    def __call__(self, y_or_y_pred: str) -> Any:
        pass

class NoopPostProcessor(PostProcessor):
    def __call__(self, y_or_y_pred: str) -> str:
        return y_or_y_pred
    
class ClassificationPostProcessor(PostProcessor):
    _label2id: Dict[str, int]
    _unk_id: int

    def __init__(self, label2id: Dict[str, int], unk_id=-1):
        self._label2id = label2id
        self._unk_id = unk_id
        
    def __call__(self, y_or_y_pred: str) -> int:
        try:
            return self._label2id[y_or_y_pred]
        except KeyError:
            return self._unk_id