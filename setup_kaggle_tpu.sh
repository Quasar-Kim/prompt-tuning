#!/bin/bash
curl -sSL https://install.python-poetry.org | python3 -
~/.local/bin/poetry env use $(which python3.8)
~/.local/bin/poetry install -q --no-root --with xla
~/.local/bin/poetry remove lightning
git clone https://github.com/lightning-ai/lightning.git
cd lightning
git checkout 83f6832
cd ..
~/.local/bin/poetry add "./lightning"[extra]
mkdir /kaggle/working/prompt-tuning/data/khs
cp /kaggle/input/kcmoe-pretrain-finetune-dataset/hate-speech/*.parquet /kaggle/working/prompt-tuning/data/khs
