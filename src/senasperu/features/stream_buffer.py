"""Buffer de ventana deslizante para la inferencia en tiempo real.

Acumula los landmarks **crudos** de los últimos segundos y, cada cierto número
de frames, entrega la ventana lista para el modelo.

Detalle importante: la ventana se normaliza entera, con el mismo
:class:`~senasperu.features.normalize.LandmarkNormalizer` que se usó al
entrenar. Normalizar frame a frame sería más barato, pero produciría valores
ligeramente distintos a los del entrenamiento —la interpolación de huecos mira
frames vecinos— y esa diferencia se paga en precisión.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from senasperu.config import Config
from senasperu.features.landmarks import HolisticResult
from senasperu.features.normalize import LandmarkNormalizer
from senasperu.features.vector import FeatureLayout, to_feature_vector
from senasperu.features.windows import resample_sequence


class StreamWindowBuffer:
    """Ventana deslizante sobre el flujo de landmarks en vivo."""

    def __init__(
        self,
        layout: FeatureLayout,
        normalizer: LandmarkNormalizer,
        *,
        window_seconds: float,
        frames_per_window: int,
        stride_frames: int,
        fps: float,
    ) -> None:
        """Args:
        layout: Layout del vector crudo.
        normalizer: Normalizador, el mismo que se usa en entrenamiento.
        window_seconds: Segundos reales que abarca la ventana.
        frames_per_window: Frames a los que se remuestrea antes del modelo.
        stride_frames: Cada cuántos frames nuevos se produce una ventana.
        fps: FPS esperados de la cámara, para dimensionar el buffer.
        """
        self._layout = layout
        self._normalizer = normalizer
        self._frames_per_window = int(frames_per_window)
        self._stride = max(1, int(stride_frames))
        self._capacity = max(2, int(round(window_seconds * max(1.0, fps))))
        self._frames: deque[np.ndarray] = deque(maxlen=self._capacity)
        self._since_last = 0

    @classmethod
    def from_config(
        cls, config: Config, layout: FeatureLayout, normalizer: LandmarkNormalizer
    ) -> StreamWindowBuffer:
        """Construye el buffer con las secciones ``ventana`` y ``camara``."""
        return cls(
            layout,
            normalizer,
            window_seconds=float(config.require("ventana.duracion_segundos")),
            frames_per_window=int(config.require("ventana.frames_por_ventana")),
            stride_frames=int(config.require("ventana.paso_frames")),
            fps=float(config.require("camara.fps_objetivo")),
        )

    @property
    def capacity(self) -> int:
        """Frames crudos que caben en la ventana."""
        return self._capacity

    @property
    def filled(self) -> int:
        """Frames acumulados hasta ahora."""
        return len(self._frames)

    @property
    def ready_ratio(self) -> float:
        """Cuánto falta para tener la primera ventana completa (0.0-1.0)."""
        return min(1.0, self.filled / self._capacity)

    def push(self, result: HolisticResult) -> np.ndarray | None:
        """Agrega un frame y devuelve una ventana si toca inferir.

        Args:
            result: Landmarks del frame recién procesado.

        Returns:
            Matriz ``(frames_por_ventana, features)`` lista para el modelo, o
            ``None`` si todavía no corresponde inferir.
        """
        self._frames.append(to_feature_vector(result, self._layout))
        self._since_last += 1

        if len(self._frames) < self._capacity or self._since_last < self._stride:
            return None

        self._since_last = 0
        crudos = np.stack(self._frames)
        secuencia = self._normalizer.normalize(crudos)
        return resample_sequence(secuencia.features, self._frames_per_window)

    def clear(self) -> None:
        """Vacía el buffer (al reiniciar la conversación o la cámara)."""
        self._frames.clear()
        self._since_last = 0
