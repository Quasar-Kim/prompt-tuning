#!/bin/bash
if command -v "sudo" >/dev/null 2>&1; then
    sudo apt-get install -y python3.8
else
    apt-get install -y python3.8
fi
curl -sSL https://install.python-poetry.org | python3 -
~/.local/bin/poetry env use $(which python3.8)
~/.local/bin/poetry install -q --no-root