from typing import Protocol, TYPE_CHECKING
from t2tpipe.dataclass import Env


class SetupMixin:
    _env: Env

    def setup(self, env: Env):
        self._env = env
