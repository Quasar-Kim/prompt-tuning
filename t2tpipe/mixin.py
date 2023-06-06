from typing import TYPE_CHECKING, Protocol

from t2tpipe.dataclass import Env


class SetupMixin:
    _env: Env

    def setup(self, env: Env):
        self._env = env
