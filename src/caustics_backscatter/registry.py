"""Name -> constructor registries so every swappable component is selected from config.

A config section such as ``{"name": "smp_dense", "arch": "DeepLabV3Plus"}`` is built
with ``MODELS.build(section)``: ``name`` picks the registered class and the remaining
keys become constructor kwargs.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from omegaconf import DictConfig, OmegaConf


class Registry:
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, Callable[..., Any]] = {}

    def register(self, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(obj: Callable[..., Any]) -> Callable[..., Any]:
            if name in self._items:
                raise KeyError(f"{self.kind} '{name}' is already registered")
            self._items[name] = obj
            return obj

        return decorator

    def get(self, name: str) -> Callable[..., Any]:
        if name not in self._items:
            raise KeyError(f"Unknown {self.kind} '{name}'. Registered: {sorted(self._items)}")
        return self._items[name]

    def build(self, section: Mapping[str, Any] | DictConfig, /, **extra: Any) -> Any:
        if isinstance(section, DictConfig):
            section = OmegaConf.to_container(section, resolve=True)
        kwargs = dict(section)
        name = kwargs.pop("name")
        return self.get(name)(**kwargs, **extra)

    def names(self) -> list[str]:
        return sorted(self._items)


MODELS = Registry("model")
GEOMETRY_PROVIDERS = Registry("geometry provider")
PIXEL_WEIGHTERS = Registry("pixel weighter")
TASKS = Registry("task")
EVALUATORS = Registry("evaluator")
