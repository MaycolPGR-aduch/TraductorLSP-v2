"""Extracción de features: MediaPipe Holistic, landmarks y dibujo del esqueleto.

Los símbolos que dependen de OpenCV o MediaPipe se importan de forma diferida,
para que los módulos de lógica pura sigan siendo testeables sin esas librerías.
"""

from __future__ import annotations

__all__ = [
    "HolisticExtractor",
    "HolisticResult",
    "HolisticThread",
    "LandmarkOverlay",
    "ProcessedFrame",
]

_LAZY = {
    "HolisticResult": "senasperu.features.landmarks",
    "HolisticExtractor": "senasperu.features.holistic",
    "LandmarkOverlay": "senasperu.features.overlay",
    "HolisticThread": "senasperu.features.holistic_thread",
    "ProcessedFrame": "senasperu.features.holistic_thread",
}


def __getattr__(name: str):  # noqa: D103 - import diferido
    modulo = _LAZY.get(name)
    if modulo is None:
        raise AttributeError(f"El módulo 'senasperu.features' no tiene el atributo '{name}'.")
    from importlib import import_module

    return getattr(import_module(modulo), name)
