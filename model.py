from functools import partial

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from transformers import GPT2LMHeadModel, T5ForConditionalGeneration

import tokenizer
from t2tpipe import Model, datapipe, feature_converter, postprocessor
from t2tpipe.base import BaseLightningModule
from t2tpipe.dataclass import (
    DecSampleForPrediction,
    DecSampleForTrain,
    EncDecSampleForPrediction,
    EncDecSampleForTrain,
    Env,
    ModelPredictionOutput,
    ModelTrainOutput,
)


class InvSqrtScheduler(LambdaLR):
    def __init__(self, optimizer, warmup_epochs: int):
        self._warmup_epochs = warmup_epochs
        super().__init__(optimizer, lr_lambda=self._lr_lambda)

    def _lr_lambda(self, epoch):
        if epoch < self._warmup_epochs:
            return epoch / self._warmup_epochs
        return (epoch**-0.5) * (self._warmup_epochs**0.5)


class KeT5Module(BaseLightningModule):
    _model_name: str
    _use_lora: bool

    def __init__(self, model_name: str, use_lora: bool = False):
        super().__init__()
        self._model_name = model_name
        self._use_lora = use_lora

    def configure(self, env: Env):
        super().configure(env)
        self.module = T5ForConditionalGeneration.from_pretrained(self._model_name)
        if self._use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                inference_mode=False,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
            )
            self.module = get_peft_model(self.module, lora_config)

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
        from torch_xla.distributed.fsdp.wrap import (  # type: ignore
            transformer_auto_wrap_policy,
        )
        from transformers.models.t5.modeling_t5 import T5Block

        return partial(transformer_auto_wrap_policy, transformer_layer_cls={T5Block})


ket5_small = Model(
    name="ket5_small",
    feature_converter=feature_converter.EncDecFeatureConverter(),
    module=KeT5Module("KETI-AIR/ke-t5-small"),
    tokenizer=tokenizer.KeT5Tokenizer(),
)

ket5_base = Model(
    name="ket5_base",
    feature_converter=feature_converter.EncDecFeatureConverter(),
    module=KeT5Module("KETI-AIR/ke-t5-base"),
    tokenizer=tokenizer.KeT5Tokenizer(),
)

ket5_base_lora = Model(
    name="ket5_base_lora",
    feature_converter=feature_converter.EncDecFeatureConverter(),
    module=KeT5Module("KETI-AIR/ke-t5-base", use_lora=True),
    tokenizer=tokenizer.KeT5Tokenizer(),
)

ket5_large = Model(
    name="ket5_large",
    feature_converter=feature_converter.EncDecFeatureConverter(),
    module=KeT5Module("KETI-AIR/ke-t5-large"),
    tokenizer=tokenizer.KeT5Tokenizer(),
)

ket5_large_lora = Model(
    name="ket5_large_lora",
    feature_converter=feature_converter.EncDecFeatureConverter(),
    module=KeT5Module("KETI-AIR/ke-t5-large", use_lora=True),
    tokenizer=tokenizer.KeT5Tokenizer(),
)


class KoGpt2Module(BaseLightningModule):
    def configure(self, env: Env):
        super().configure(env)
        self.module = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")

    def forward(self, batch: DecSampleForTrain):
        # TODO: batch.y가 그대로 사용되는지, 아니면 shift되는지 확인해보기
        return self.module(
            input_ids=batch.x,
            attention_mask=batch.attention_mask,
            labels=batch.y,
        )  # type: ignore

    def _step_train(self, batch: DecSampleForTrain) -> ModelTrainOutput:
        output = self.module(
            input_ids=batch.x,
            attention_mask=batch.attention_mask,
            labels=batch.y,
        )  # type: ignore
        y_pred = torch.argmax(output.logits, dim=-1)
        return ModelTrainOutput(x=batch.x, y=batch.y, y_pred=y_pred, loss=output.loss)

    def _step_prediction(self, batch: DecSampleForPrediction) -> ModelPredictionOutput:
        eos_token_id = torch.tensor(
            self._env.model.tokenizer.eos_token_id, device=batch.x.device
        )
        y_preds = []
        # run sample by sample
        assert batch.attention_mask is not None
        for x, attention_mask in zip(batch.x, batch.attention_mask):
            while True:
                output = self.module(
                    input_ids=x,
                    attention_mask=attention_mask,
                )  # type: ignore
                y_pred = torch.argmax(output.logits, dim=-1)  # shape: (N,)
                if torch.equal(y_pred[-1], eos_token_id):
                    y_preds.append(y_pred)
                    break
        max_len = max(len(y_pred) for y_pred in y_preds)
        pad_token_id = self._env.model.tokenizer.pad_token_id
        assert pad_token_id is not None
        y_preds = [
            torch.nn.functional.pad(
                y_pred, (0, max_len - y_pred.size(0)), value=pad_token_id
            )
            for y_pred in y_preds
        ]
        return ModelPredictionOutput(
            x=batch.x,
            y_pred=torch.stack(y_preds, dim=0),
        )

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
    pipes={
        "prefix_adder": datapipe.PrefixAdder(x_prefix="문장: ", y_prefix="분류: "),
    },
    postprocessors={
        "lm_output_slicer": postprocessor.LMOutputSlicer(output_prefix="분류: "),
    },
)
