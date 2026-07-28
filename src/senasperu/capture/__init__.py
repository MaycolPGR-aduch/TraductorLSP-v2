"""Captura de video: fuentes de frames, cola de descarte e hilo dedicado.

Los símbolos se importan de forma diferida para que la lógica pura (por ejemplo
:class:`~senasperu.capture.frame_queue.DropOldestQueue`) pueda usarse y testearse
sin tener OpenCV instalado.
"""

from __future__ import annotations

from senasperu.capture.frame_queue import DropOldestQueue

__all__ = [
    "CameraError",
    "CameraSource",
    "CaptureThread",
    "DropOldestQueue",
    "Frame",
    "FrameSource",
    "VideoFileSource",
    "create_frame_source",
]

_LAZY = {
    "CameraError": "senasperu.capture.frame_source",
    "CameraSource": "senasperu.capture.frame_source",
    "Frame": "senasperu.capture.frame_source",
    "FrameSource": "senasperu.capture.frame_source",
    "VideoFileSource": "senasperu.capture.frame_source",
    "create_frame_source": "senasperu.capture.frame_source",
    "CaptureThread": "senasperu.capture.capture_thread",
}


def __getattr__(name: str):  # noqa: D103 - import diferido (requiere OpenCV)
    modulo = _LAZY.get(name)
    if modulo is None:
        raise AttributeError(f"El módulo 'senasperu.capture' no tiene el atributo '{name}'.")
    from importlib import import_module

    return getattr(import_module(modulo), name)
