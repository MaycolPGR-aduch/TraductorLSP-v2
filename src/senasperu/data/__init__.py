"""Dataset: control de calidad, acumulación de repeticiones y escritura en disco.

Los símbolos que dependen de OpenCV se importan de forma diferida.
"""

from __future__ import annotations

from senasperu.data.quality import QualityChecker, QualityReport
from senasperu.data.recording import RecordingBuffer, RecordingSample

__all__ = [
    "DatasetSaveWorker",
    "DatasetWriter",
    "QualityChecker",
    "QualityReport",
    "RecordingBuffer",
    "RecordingSample",
    "SaveResult",
    "SavedRecording",
]

_LAZY = {
    "DatasetWriter": "senasperu.data.dataset_writer",
    "SavedRecording": "senasperu.data.dataset_writer",
    "DatasetSaveWorker": "senasperu.data.save_worker",
    "SaveResult": "senasperu.data.save_worker",
}


def __getattr__(name: str):  # noqa: D103 - import diferido
    modulo = _LAZY.get(name)
    if modulo is None:
        raise AttributeError(f"El módulo 'senasperu.data' no tiene el atributo '{name}'.")
    from importlib import import_module

    return getattr(import_module(modulo), name)
