import t2tpipe
from t2tpipe.util import benchmark_datamodule
from task import khs
from model import keT5Small

model, dm = t2tpipe.setup(
    model=keT5Small,
    task=khs,
    runtime_config={
        'num_warmup_epochs': 5,
        'lr': 1e-3,
        'batch_size': 16,
        'num_workers': 4,
        'is_gpu': False
    }
)

benchmark_datamodule(dm)
