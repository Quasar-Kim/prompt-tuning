from typing import Optional

import torch

from t2tpipe.datapipe import TransformDataPipe, PadderForEncDecModel
from t2tpipe.dataclass import EncDecSampleForTrain

class FeatureConverter(TransformDataPipe):
    def get_default_padder(self) -> Optional[TransformDataPipe]:
        return None

class NoopFeatureConverter(FeatureConverter):
    def __iter__(self):
        yield from iter(self.source_dp)

class EncDecFeatureConverter(FeatureConverter):
    def __iter__(self):
        tokenizer = self._env.model.tokenizer
        assert tokenizer.bos_token_id is not None
        assert tokenizer.eos_token_id is not None
        assert tokenizer.pad_token_id is not None
        for sample in iter(self.source_dp):
            yield EncDecSampleForTrain(
                enc_x=torch.cat([
                    sample.x,
                    torch.tensor([tokenizer.eos_token_id])
                ]),
                dec_x=torch.cat([
                    torch.tensor([tokenizer.bos_token_id]),
                    sample.y,
                    torch.tensor([tokenizer.eos_token_id])
                ]),
                y=torch.cat([
                    sample.y,
                    torch.tensor([tokenizer.eos_token_id]),
                    torch.tensor([tokenizer.pad_token_id])
                ])
            )

    def get_default_padder(self):
        return PadderForEncDecModel()