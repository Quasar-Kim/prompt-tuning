from typing import Optional

import torch

from t2tpipe.dataclass import (
    DecSampleForPrediction,
    DecSampleForTrain,
    EncDecSampleForPrediction,
    EncDecSampleForTrain,
    EncodedSampleForPrediction,
    EncodedSampleForTrain,
)
from t2tpipe.datapipe import PadderForDecModel, PadderForEncDecModel, TransformDataPipe


class FeatureConverter(TransformDataPipe):
    def get_default_padder(self) -> Optional[TransformDataPipe]:
        return None


class NoopFeatureConverter(FeatureConverter):
    def __iter__(self):
        yield from iter(self.source_dp)


class EncDecFeatureConverter(FeatureConverter):
    def __iter__(self):
        if self._env.prediction:
            yield from self._prediction_iter()
        else:
            yield from self._train_iter()

    def _train_iter(self):
        tokenizer = self._env.model.tokenizer
        assert tokenizer.bos_token_id is not None
        assert tokenizer.eos_token_id is not None
        assert tokenizer.pad_token_id is not None
        for sample in iter(self.source_dp):
            sample: EncodedSampleForTrain
            yield EncDecSampleForTrain(
                enc_x=torch.cat([sample.x, torch.tensor([tokenizer.eos_token_id])]),
                dec_x=torch.cat(
                    [
                        torch.tensor([tokenizer.bos_token_id]),
                        sample.y,
                        torch.tensor([tokenizer.eos_token_id]),
                    ]
                ),
                y=torch.cat(
                    [
                        sample.y,
                        torch.tensor([tokenizer.eos_token_id]),
                        torch.tensor([tokenizer.pad_token_id]),
                    ]
                ),
            )

    def _prediction_iter(self):
        tokenizer = self._env.model.tokenizer
        assert tokenizer.bos_token_id is not None
        assert tokenizer.eos_token_id is not None
        for sample in iter(self.source_dp):
            sample: EncodedSampleForPrediction
            yield EncDecSampleForPrediction(
                enc_x=torch.cat([sample.x, torch.tensor([tokenizer.eos_token_id])]),
                dec_x=torch.tensor([tokenizer.bos_token_id]),
            )

    def get_default_padder(self):
        return PadderForEncDecModel()


class DecFeatureConverter(FeatureConverter):
    """
    Dercoder-only 모델의 입력으로 사용할 수 있도록
    Language Modeling objective 형태로 샘플 변환

    훈련 샘플 예시:
        - 입력: EncodedSampleForTrain(
                    x=torch.Tensor([1, 2, 3]),
                    y=torch.Tensor([4, 5, 6]),
                )
        - 출력: DecSampleForTrain(
                    x=torch.Tensor([BOS_TOKEN, 1, 2, 3, 4, 5, 6, EOS_TOKEN]),
                    y=torch.Tensor([1, 2, 3, 4, 5, 6, EOS_TOKEN, PAD_TOKEN]),
                )

    예측 샘플 예시:
        - 입력: EncodedSampleForPrediction(
                    x=torch.Tensor([1, 2, 3])
                )
        - 출력: DecSampleForPrediction(
                    x=torch.Tensor([BOS_TOKEN, 1, 2, 3])
                )

    훈련 샘플 텍스트 예시 (실제 입력으로는 사용 불가):
        - 입력: {"x": "안녕", "y": "세상"}
        - 출력: {"x": "<s>안녕 세상</s>", "y": "안녕 세상</s><pad>"}

    예측 샘플 텍스트 예시 (실제 입력으로는 사용 불가):
        - 입력: {"x": "안녕"}
        - 출력: {"x": "<s>안녕"}
    """

    def __iter__(self):
        if self._env.prediction:
            yield from self._prediction_iter()
        else:
            yield from self._train_iter()

    def _train_iter(self):
        tokenizer = self._env.model.tokenizer
        assert tokenizer.bos_token_id is not None
        assert tokenizer.eos_token_id is not None
        assert tokenizer.pad_token_id is not None
        for sample in iter(self.source_dp):
            sample: EncodedSampleForTrain
            yield DecSampleForTrain(
                x=torch.cat(
                    [
                        torch.tensor([tokenizer.bos_token_id]),
                        sample.x,
                        sample.y,
                        torch.tensor([tokenizer.eos_token_id]),
                    ]
                ),
                y=torch.cat(
                    [
                        sample.x,
                        sample.y,
                        torch.tensor([tokenizer.eos_token_id, tokenizer.pad_token_id]),
                    ]
                ),
            )

    def _prediction_iter(self):
        tokenizer = self._env.model.tokenizer
        assert tokenizer.bos_token_id is not None
        assert tokenizer.eos_token_id is not None
        for sample in iter(self.source_dp):
            yield DecSampleForPrediction(
                x=torch.cat([torch.tensor([tokenizer.bos_token_id]), sample.x])
            )

    def get_default_padder(self):
        return PadderForDecModel()
