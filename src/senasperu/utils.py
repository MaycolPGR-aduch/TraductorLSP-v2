"""Utilidades pequeñas compartidas entre módulos (sin dependencias pesadas)."""

from __future__ import annotations

import time
from collections import deque


class FpsMeter:
    """Medidor de FPS por media móvil sobre los últimos N intervalos.

    Es más estable que ``1 / dt`` instantáneo y no acumula sesgo, lo que importa
    para validar los criterios de aceptación (≥25 FPS sostenidos).
    """

    def __init__(self, window: int = 30) -> None:
        """Args:
        window: Cantidad de intervalos considerados en la media móvil.
        """
        self._timestamps: deque[float] = deque(maxlen=max(2, window))

    def tick(self, timestamp: float | None = None) -> None:
        """Registra la ocurrencia de un frame."""
        self._timestamps.append(time.perf_counter() if timestamp is None else timestamp)

    @property
    def fps(self) -> float:
        """FPS promedio en la ventana; ``0.0`` si aún no hay datos suficientes."""
        if len(self._timestamps) < 2:
            return 0.0
        transcurrido = self._timestamps[-1] - self._timestamps[0]
        if transcurrido <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / transcurrido

    def reset(self) -> None:
        """Descarta las mediciones acumuladas."""
        self._timestamps.clear()


class LatencyMeter:
    """Media móvil de duraciones en milisegundos (por ejemplo, tiempo de MediaPipe)."""

    def __init__(self, window: int = 30) -> None:
        self._samples: deque[float] = deque(maxlen=max(1, window))

    def add(self, seconds: float) -> None:
        """Agrega una muestra expresada en segundos."""
        self._samples.append(seconds * 1000.0)

    @property
    def milliseconds(self) -> float:
        """Promedio en milisegundos; ``0.0`` si no hay muestras."""
        if not self._samples:
            return 0.0
        return sum(self._samples) / len(self._samples)

    def reset(self) -> None:
        """Descarta las mediciones acumuladas."""
        self._samples.clear()
