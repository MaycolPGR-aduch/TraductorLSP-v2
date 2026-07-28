"""Interfaz gráfica (PySide6). No toca OpenCV ni MediaPipe directamente."""

from __future__ import annotations

__all__ = ["PipelineBridge", "PipelineStats", "SmokeWindow"]


def __getattr__(name: str):  # noqa: D103 - import diferido: Qt tarda en cargar
    if name in ("PipelineBridge", "PipelineStats"):
        from senasperu.ui import pipeline_bridge

        return getattr(pipeline_bridge, name)
    if name == "SmokeWindow":
        from senasperu.ui.smoke_window import SmokeWindow

        return SmokeWindow
    raise AttributeError(f"El módulo 'senasperu.ui' no tiene el atributo '{name}'.")
