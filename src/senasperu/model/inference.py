"""Wrapper de inferencia con ONNX Runtime.

Es la única puerta del modelo hacia la aplicación final: aquí no se importa
PyTorch. Se limita el número de hilos a propósito, porque en una laptop modesta
dejar que ONNX Runtime use todos los núcleos compite con la captura de video y
termina bajando los FPS en vez de subirlos.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from senasperu.config import Config

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Prediction:
    """Resultado de clasificar una ventana.

    Attributes:
        class_index: Índice de la clase más probable.
        confidence: Probabilidad de esa clase (0.0-1.0).
        probabilities: Vector completo de probabilidades.
    """

    class_index: int
    confidence: float
    probabilities: np.ndarray


class SignClassifier:
    """Clasificador de señas sobre un modelo ONNX cuantizado."""

    def __init__(self, model_path: str | Path, *, threads: int = 2) -> None:
        """Args:
        model_path: Ruta del ``.onnx``.
        threads: Hilos que puede usar ONNX Runtime.

        Raises:
            FileNotFoundError: Si el modelo no existe.
        """
        import onnxruntime as ort

        ruta = Path(model_path)
        if not ruta.is_file():
            raise FileNotFoundError(
                f"No se encontró el modelo en {ruta}. Entrénalo y expórtalo con "
                "'python scripts/train.py' y 'python scripts/export_onnx.py'."
            )

        opciones = ort.SessionOptions()
        opciones.intra_op_num_threads = max(1, threads)
        opciones.inter_op_num_threads = 1
        opciones.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            str(ruta), sess_options=opciones, providers=["CPUExecutionProvider"]
        )
        entrada = self._session.get_inputs()[0]
        self._input_name = entrada.name
        self._input_shape = entrada.shape
        logger.info("Modelo ONNX cargado: %s (entrada %s)", ruta.name, self._input_shape)

    @classmethod
    def from_config(cls, config: Config) -> SignClassifier:
        """Construye el clasificador con la sección ``inferencia`` del YAML."""
        return cls(
            config.resolve_path("inferencia.ruta_modelo_onnx"),
            threads=int(config.get("inferencia.hilos_onnx", 2)),
        )

    @property
    def frames_per_window(self) -> int:
        """Frames que espera el modelo por ventana."""
        return int(self._input_shape[1])

    @property
    def input_size(self) -> int:
        """Largo del vector de features por frame."""
        return int(self._input_shape[2])

    def predict(self, window: np.ndarray) -> Prediction:
        """Clasifica una ventana.

        Args:
            window: Matriz ``(frames, features)`` o ``(1, frames, features)``.

        Returns:
            La predicción, con probabilidades ya normalizadas.
        """
        lote = window[None, ...] if window.ndim == 2 else window
        logits = self._session.run(None, {self._input_name: lote.astype(np.float32)})[0]
        probabilidades = _softmax(logits[0])
        indice = int(np.argmax(probabilidades))
        return Prediction(
            class_index=indice,
            confidence=float(probabilidades[indice]),
            probabilities=probabilidades,
        )


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Softmax numéricamente estable."""
    desplazado = logits - np.max(logits)
    exponenciales = np.exp(desplazado)
    return (exponenciales / np.sum(exponenciales)).astype(np.float32)
