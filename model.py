from functools import partial

from transformers import T5ForConditionalGeneration, GPT2LMHeadModel
import torch
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from peft import get_peft_model, LoraConfig, TaskType

from t2tpipe import feature_converter, postprocessor, Model
from t2tpipe.base import BaseLightningModule
from t2tpipe.dataclass import (
    EncDecSampleForTrain,
    EncDecSampleForPrediction,
    DecSampleForTrain,
    DecSampleForPrediction,
    ModelPredictionOutput,
    ModelTrainOutput,
)
import tokenizer


class InvSqrtScheduler(LambdaLR):
    def __init__(self, optimizer, warmup_epochs: int):
        self._warmup_epochs = warmup_epochs
        super().__init__(optimizer, lr_lambda=self._lr_lambda)

    def _lr_lambda(self, epoch):
        if epoch < self._warmup_epochs:
            return epoch / self._warmup_epochs
        return (epoch**-0.5) * (self._warmup_epochs**0.5)


class KeT5Module(BaseLightningModule):
    def __init__(self, model_name: str):
        super().__init__()
        self.module = T5ForConditionalGeneration.from_pretrained(model_name)

    def forward(self, batch: EncDecSampleForTrain):
        return self.module(
            input_ids=batch.enc_x,
            attention_mask=batch.enc_attention_mask,
            decoder_input_ids=batch.dec_x,
            decoder_attention_mask=batch.dec_attention_mask,
            labels=batch.y,
        )  # type: ignore

    def _step_train(self, batch: EncDecSampleForTrain) -> ModelTrainOutput:
        output = self(batch)
        y_pred = torch.argmax(output.logits, dim=-1)
        return ModelTrainOutput(
            x=batch.enc_x, y=batch.y, y_pred=y_pred, loss=output.loss
        )

    def _step_prediction(
        self, batch: EncDecSampleForPrediction
    ) -> ModelPredictionOutput:
        raise NotImplementedError()

    def configure_optimizers(self):
        cfg = self._env.runtime_config
        assert "lr" in cfg
        assert "num_warmup_epochs" in cfg
        optimizer = optim.Adam(self.parameters(), lr=cfg["lr"])
        lr_scheduler = InvSqrtScheduler(
            optimizer, warmup_epochs=cfg["num_warmup_epochs"]
        )
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    @property
    def xla_fsdp_auto_wrap_policy(self):
        from torch_xla.distributed.fsdp.wrap import transformer_auto_wrap_policy  # type: ignore
        from transformers.models.t5.modeling_t5 import T5Block

        return partial(transformer_auto_wrap_policy, transformer_layer_cls={T5Block})


class KeT5SmallModule(KeT5Module):
    def __init__(self):
        super().__init__("KETI-AIR/ke-t5-small")


class KeT5BaseModule(KeT5Module):
    def __init__(self):
        super().__init__("KETI-AIR/ke-t5-base")


class KeT5LargeModule(KeT5Module):
    def __init__(self):
        super().__init__("KETI-AIR/ke-t5-large")


def apply_lora(module: BaseLightningModule):
    assert hasattr(module, "module")
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    module.module = get_peft_model(module.module, lora_config)
    return module


ket5_small = Model(
    name="ket5_small",
    feature_converter=feature_converter.EncDecFeatureConverter(),
    module=KeT5SmallModule(),
    tokenizer=tokenizer.KeT5Tokenizer(),
)

ket5_base = Model(
    name="ket5_base",
    feature_converter=feature_converter.EncDecFeatureConverter(),
    module=KeT5BaseModule(),
    tokenizer=tokenizer.KeT5Tokenizer(),
)

ket5_base_lora = Model(
    name="ket5_base_lora",
    feature_converter=feature_converter.EncDecFeatureConverter(),
    module=apply_lora(KeT5BaseModule()),
    tokenizer=tokenizer.KeT5Tokenizer(),
)

ket5_large = Model(
    name="ket5_large",
    feature_converter=feature_converter.EncDecFeatureConverter(),
    module=KeT5SmallModule(),
    tokenizer=tokenizer.KeT5Tokenizer(),
)

ket5_large_lora = Model(
    name="ket5_large_lora",
    feature_converter=feature_converter.EncDecFeatureConverter(),
    module=apply_lora(KeT5SmallModule()),
    tokenizer=tokenizer.KeT5Tokenizer(),
)


class KoGpt2Module(BaseLightningModule):
    def __init__(self):
        self.module = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")

    def forward(self, batch: DecSampleForTrain):
        # TODO: batch.y가 그대로 사용되는지, 아니면 shift되는지 확인해보기
        return self.module(
            input_ids=batch.x,
            attention_mask=batch.attention_mask,
            labels=batch.y,
        )  # type: ignore

    def _step_train(self, batch: DecSampleForTrain) -> ModelTrainOutput:
        output = self(batch)
        y_pred = torch.argmax(output.logits, dim=-1)
        return ModelTrainOutput(x=batch.x, y=batch.y, y_pred=y_pred, loss=output.loss)

    def _step_prediction(self, batch: DecSampleForPrediction) -> ModelPredictionOutput:
        raise NotImplementedError()

    def configure_optimizers(self):
        cfg = self._env.runtime_config
        assert "lr" in cfg
        assert "num_warmup_epochs" in cfg
        optimizer = optim.Adam(self.parameters(), lr=cfg["lr"])
        lr_scheduler = InvSqrtScheduler(
            optimizer, warmup_epochs=cfg["num_warmup_epochs"]
        )
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}


kogpt2 = Model(
    name="kogpt2",
    feature_converter=feature_converter.DecFeatureConverter(),
    module=KoGpt2Module(),
    tokenizer=tokenizer.KoGpt2Tokenizer(),
    postprocessors={
        "before_decoder": postprocessor.LMOutputSlicer(),
    },
)
