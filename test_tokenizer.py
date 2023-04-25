import torch
from tokenizer import KeT5Tokenizer

class TestKeT5Tokenizer:
    def test_encode_train_sample(self):
        tokenizer = KeT5Tokenizer()
        x = '얼어붙은 마음에 누가 입맞춰 줄까요'
        y = '그 말의 근거가 될수 있나요'
        inp = tokenizer.encode_train_sample(x, y, pad=False)
        attention_mask = torch.tensor(inp['attention_mask'])
        decoder_attention_mask = torch.tensor(inp['decoder_attention_mask'])
        assert tokenizer.decode(inp['input_ids']) == '얼어붙은 마음에 누가 입맞춰 줄까요</s>'
        assert torch.equal(torch.ones_like(attention_mask), attention_mask)
        assert tokenizer.decode(inp['decoder_input_ids']) == '<s>그 말의 근거가 될수 있나요</s>'
        assert torch.equal(torch.ones_like(decoder_attention_mask), decoder_attention_mask)
        assert tokenizer.decode(inp['y']) == '그 말의 근거가 될수 있나요</s>'

