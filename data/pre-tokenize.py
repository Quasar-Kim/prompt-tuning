from pathlib import Path
import pickle
import pandas
import torch
from tokenizer import KeT5Tokenizer

tokenizer = KeT5Tokenizer()

def pre_tokenize_khs(set):
    root = Path('data/khs')
    if set == 'train':
        p = root / 'train.parquet'
        p_out = root / 'train.pt'
    elif set == 'validation':
        p = root / 'validation.parquet'
        p_out = root / 'validation.pt'
    else:
        raise NotImplementedError()
    df = pandas.read_parquet(p)
    samples = []
    for _, row in df.iterrows():
        x = row['text']
        y = dict(hate='혐오', offensive='공격적', none='일반적')[row['label']]
        sample = tokenizer.encode_train_sample(x, y, pad_to=128)
        samples.append({k: torch.tensor(v) for k, v in sample.items()})
    torch.save(samples, p_out)

# run with python -m data.pre-tokenize
pre_tokenize_khs('train')
pre_tokenize_khs('validation')
    