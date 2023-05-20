#!/bin/bash
# install python3.8
if ! command -v "python3.8" >/dev/null 2>&1; then
    if command -v "sudo" >/dev/null 2>&1; then
        sudo apt-get install -y python3.8
    else
        apt-get install -y python3.8
    fi   
fi

# install dependencies
(curl -sSL https://install.python-poetry.org | python3 -) > /dev/null
~/.local/bin/poetry env use $(which python3.8)
~/.local/bin/poetry config installer.max-workers 10
~/.local/bin/poetry install -q --no-root --with xla

# install lightning 2.1.0.dev
~/.local/bin/poetry remove -q lightning
git clone --quiet https://github.com/lightning-ai/lightning.git
cd lightning
git checkout --quiet 83f6832
cd ..
~/.local/bin/poetry add -q "./lightning"[extra]

# copy parquet files to repository
mkdir -p /kaggle/working/prompt-tuning/data/khs
cp /kaggle/input/kcmoe-pretrain-finetune-dataset/hate-speech/*.parquet /kaggle/working/prompt-tuning/data/khs

mkdir -p /kaggle/working/prompt-tuning/data/nsmc
cp /kaggle/input/nsmc-parquet/*.parquet /kaggle/working/prompt-tuning/data/nsmc
