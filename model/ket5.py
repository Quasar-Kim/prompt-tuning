from functools import partial
import torch
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from transformers import T5ForConditionalGeneration

def inv_sqrt_lambda(epoch, warmup_epochs):
    if epoch < warmup_epochs:
        return epoch / warmup_epochs
    return (epoch ** -0.5) * (warmup_epochs ** 0.5)

# TODO: add customizable metric
class KeT5(LightningModule):
    def __init__(self, *, variant, tokenizer, lr, num_warmup_epochs, **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=['variant', 'tokenizer'])
        self.model = T5ForConditionalGeneration.from_pretrained(f'KETI-AIR/ke-t5-{variant}')
        self.tokenizer = tokenizer
        self.train_losses = []
        self.validation_losses = []

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        y_true = batch.pop('y')
        logits = self(**batch).logits
        loss = self.loss(logits, y_true)
        self.train_losses.append(loss)
        return { 'loss': loss }
    
    def validation_step(self, batch, batch_idx):
        y_true = batch.pop('y')
        logits = self(**batch).logits
        loss = self.loss(logits, y_true)
        self.validation_losses.append(loss)
        return { 'loss': loss }
    
    def loss(self, logits, y_true):
        # NOTE: KeT5 does not have builtin loss
        logits = logits.permute(0, 2, 1) # (B, vocab_size, N)
        loss = F.cross_entropy(logits, y_true, ignore_index=self.tokenizer._tokenizer.pad_token_id)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = LambdaLR(optimizer, lr_lambda=partial(inv_sqrt_lambda, warmup_epochs=self.hparams.num_warmup_epochs))
        return { 'optimizer': optimizer, 'lr_scheduler': lr_scheduler }
    
    def on_train_epoch_end(self, *args):
        loss = torch.stack(self.train_losses).mean()
        self.train_losses.clear()
        self.log('train_loss', loss, on_epoch=True, sync_dist=True, prog_bar=True)

    def on_validation_epoch_end(self, *args):
        loss = torch.stack(self.validation_losses).mean()
        self.validation_losses.clear()
        self.log('val_loss', loss, on_epoch=True, sync_dist=True)

class KeT5Small(KeT5):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, variant='small')

class KeT5Base(KeT5):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, variant='base')

class KeT5Large(KeT5):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, variant='large')