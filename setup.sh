sudo apt-get update -y
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get install -y software-properties-common python3.11
curl -sSL https://install.python-poetry.org | python3 -
~/.local/bin/poetry config --local virtualenvs.in-project true
~/.local/bin/poetry env use $(which python3.11)
~/.local/bin/poetry install -q --no-root