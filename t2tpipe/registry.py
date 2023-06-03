from typing import Dict, Protocol

from t2tpipe.dataclass import Task, Model


class HasName(Protocol):
    name: str


class Registry:
    _data: Dict[str, Task] = {}

    def register(self, entry: HasName):
        _data[entry.name] = entry

    def unregister(self, entry: HasName):
        _data.pop(entry.name)

    def has(self, name: str):
        return name in self._data

    def get(self, name: str):
        if not self.has(name):
            cls_name = self.__class__.__name__
            raise KeyError(f"Entry `{name}` does not exists in the {cls_name}")
        return self._data[name]


task_registry = Registry()

model_registry = Registry()
