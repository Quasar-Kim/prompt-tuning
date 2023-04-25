from data.khs import KHSDataModule
from tokenizer import KeT5Tokenizer

class TestKHSDataModule:
    def test_train_shape(self):
        tokenizer = KeT5Tokenizer()
        dm = KHSDataModule(batch_size=2, tokenizer=tokenizer)
        dm.prepare_data()
        dm.setup(stage='fit')
        loader = dm.train_dataloader()
        batch = next(iter(loader))
        assert batch['input_ids'].shape == (2, 128)
        assert batch['attention_mask'].shape == (2, 128)
        assert batch['decoder_input_ids'].shape == (2, 128)
        assert batch['decoder_attention_mask'].shape == (2, 128)
        assert batch['y'].shape == (2, 128)

    def test_validation_shape(self):
        tokenizer = KeT5Tokenizer()
        dm = KHSDataModule(batch_size=2, tokenizer=tokenizer)
        dm.prepare_data()
        dm.setup(stage='fit')
        loader = dm.val_dataloader()
        batch = next(iter(loader))
        assert batch['input_ids'].shape == (2, 128)
        assert batch['attention_mask'].shape == (2, 128)
        assert batch['decoder_input_ids'].shape == (2, 128)
        assert batch['decoder_attention_mask'].shape == (2, 128)
        assert batch['y'].shape == (2, 128)