"""Model modules, one per architecture family. Each module registers the Hugging Face `architectures`
names it implements; `load_model` picks one from config.json.

A model class implements engine_api.Model and a constructor
    from_checkpoint(ck: loader.Checkpoint, device: str, max_seq: int, act: str = "as_shipped",
                    kv: str = "as_shipped") -> Model
Add a family by dropping a module in this package that calls `register(...)`; nothing else changes.
"""
from __future__ import annotations

import importlib
import pkgutil

REGISTRY: dict[str, type] = {}


def register(*architectures: str):
    def deco(cls):
        for a in architectures:
            REGISTRY[a] = cls
        return cls
    return deco


def _discover() -> None:
    for m in pkgutil.iter_modules(__path__):
        importlib.import_module(f"{__name__}.{m.name}")


def model_class(architecture: str) -> type:
    _discover()
    if architecture not in REGISTRY:
        raise NotImplementedError(f"no model module for {architecture!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[architecture]


def load_model(model_dir: str, device: str = "cuda", max_seq: int = 4096, **opts):
    from ..loader import Checkpoint
    ck = Checkpoint(model_dir, device=device)
    arch = (ck.config.get("architectures") or [None])[0]
    return model_class(arch).from_checkpoint(ck, device=device, max_seq=max_seq, **opts)
