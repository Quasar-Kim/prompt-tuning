from data.feature_converter import EncDecFeatureConverter
from tokenizer import KeT5Tokenizer

tokenizer = KeT5Tokenizer()

def pad(l: list, n: int):
    return l + [tokenizer.pad_token_id] * (n - len(l))

class TestEncDecFeatureConverter:
    def test_no_padding(self):
        sample = {
            'x': '안녕하세요?',
            'y': 'hello?'
        }
        converter = EncDecFeatureConverter()
        converter.tokenizer = tokenizer
        converted = converter.convert(sample)
        encoded_x = tokenizer.encode(sample['x'])
        encoded_y = tokenizer.encode(sample['y'])
        assert converted['enc_x'].tolist() == encoded_x['input_ids']
        assert converted['enc_attention_mask'].tolist() == encoded_x['attention_mask']
        assert converted['dec_x'].tolist() == [tokenizer.bos_token_id] + encoded_y['input_ids']
        assert converted['dec_attention_mask'].tolist() == [1] + encoded_y['attention_mask']
        assert converted['y'].tolist() == encoded_y['input_ids'] + [tokenizer.pad_token_id]
        assert converted['actual_y_len'].item() == len(encoded_y['input_ids']) - 1

    def test_with_padding(self):
        sample = {
            'x': '안녕하세요?',
            'y': 'hello?'
        }
        pad_to = 10
        converter = EncDecFeatureConverter()
        converter.tokenizer = tokenizer
        converter.pad_to = pad_to
        converted = converter.convert(sample)
        encoded_x = tokenizer.encode(sample['x'])
        encoded_y = tokenizer.encode(sample['y'])
        assert converted['enc_x'].tolist() == pad(encoded_x['input_ids'], pad_to)
        assert converted['enc_attention_mask'].tolist() == pad(encoded_x['attention_mask'], pad_to)
        assert converted['dec_x'].tolist() == pad([tokenizer.bos_token_id] + encoded_y['input_ids'], pad_to)
        assert converted['dec_attention_mask'].tolist() == pad([1] + encoded_y['attention_mask'], pad_to)
        assert converted['y'].tolist() == pad(encoded_y['input_ids'] + [tokenizer.pad_token_id], pad_to)
        assert converted['actual_y_len'].item() == len(encoded_y['input_ids']) - 1