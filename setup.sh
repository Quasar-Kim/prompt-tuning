sudo apt update
sudo apt upgrade
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
curl -sSL https://install.python-poetry.org | python3 -
PATH="~/.local/bin:$PATH"
poetry env use $(which python3.11)
poetry install --no-root