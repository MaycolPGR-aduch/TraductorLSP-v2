"""Modelo: arquitectura (PyTorch), export a ONNX e inferencia.

Los imports son diferidos a propósito: la aplicación final solo necesita
``SignClassifier`` (ONNX Runtime) y no debe arrastrar PyTorch.
"""

from __future__ import annotations

__all__ = ["Prediction", "SignClassifier", "build_model", "count_parameters"]

_LAZY = {
    "SignClassifier": "senasperu.model.inference",
    "Prediction": "senasperu.model.inference",
    "build_model": "senasperu.model.architecture",
    "count_parameters": "senasperu.model.architecture",
}


def __getattr__(name: str):  # noqa: D103 - import diferido
    modulo = _LAZY.get(name)
    if modulo is None:
        raise AttributeError(f"El módulo 'senasperu.model' no tiene el atributo '{name}'.")
    from importlib import import_module

    return getattr(import_module(modulo), name)
