#!/bin/bash
sudo apt-get install -y python3.8
curl -sSL https://install.python-poetry.org | python3 -
~/.local/bin/poetry config --local virtualenvs.in-project true
~/.local/bin/poetry env use $(which python3.8)
~/.local/bin/poetry install -q --no-root