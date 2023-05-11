#!/bin/bash
# install python 3.8
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
~/.local/bin/poetry install -q --no-root

# copy parquet files to repository
# TODO: make this general
mkdir -p /kaggle/working/prompt-tuning/data/khs
cp /kaggle/input/kcmoe-pretrain-finetune-dataset/hate-speech/*.parquet /kaggle/working/prompt-tuning/data/khs