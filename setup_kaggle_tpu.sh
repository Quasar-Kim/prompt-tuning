#!/bin/bash
curl -sSL https://install.python-poetry.org | python3 -
~/.local/bin/poetry env use $(which python3.9)
~/.local/bin/poetry install -q --no-root