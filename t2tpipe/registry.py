from typing import Dict, Protocol

from t2tpipe.dataclass import Model, Task


class HasName(Protocol):
    name: str


class Registry:
    _data: Dict[str, HasName] = {}

    def register(self, entry: HasName):
        self._data[entry.name] = entry

    def unregister(self, entry: HasName):
        self._data.pop(entry.name)

    def has(self, name: str):
        return name in self._data

    def get(self, name: str):
        if not self.has(name):
            cls_name = self.__class__.__name__
            raise KeyError(f"Entry `{name}` does not exists in the {cls_name}")
        return self._data[name]


task_registry = Registry()

model_registry = Registry()
