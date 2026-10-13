"""YAML configs with ``_base_`` inheritance and ``key=value`` command-line overrides."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def _load_with_bases(path: Path) -> DictConfig:
    cfg = OmegaConf.load(path)
    base = cfg.pop("_base_", None)
    if base is None:
        return cfg
    base_cfg = _load_with_bases((path.parent / base).resolve())
    return OmegaConf.merge(base_cfg, cfg)


def load_config(path: str | Path, overrides: Sequence[str] = ()) -> DictConfig:
    cfg = _load_with_bases(Path(path).resolve())
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    OmegaConf.resolve(cfg)
    return cfg
