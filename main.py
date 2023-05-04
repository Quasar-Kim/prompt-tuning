import warnings
from lightning.pytorch.cli import LightningCLI
import model
import data
import tokenizer
import torchmetrics

if __name__ == '__main__':
    warnings.filterwarnings('ignore', r'.*does not have many workers.*')
    cli = LightningCLI()