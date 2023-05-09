#!/bin/bash
curl -sSL https://install.python-poetry.org | python3 -
~/.local/bin/poetry export -f requirements.txt > requirements.txt
pip install -r requirements.txt