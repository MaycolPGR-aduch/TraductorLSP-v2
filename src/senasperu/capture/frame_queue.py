"""Cola con política de descarte del elemento más viejo.

Regla no negociable del proyecto: **antes descartar frames que acumular retraso**.
Esta cola nunca bloquea al productor; si está llena, tira el elemento más antiguo
y encola el nuevo, de modo que el consumidor siempre trabaja sobre lo más reciente.

Es lógica pura (sin cámara ni Qt), por lo que se testea directamente con pytest.
"""

from __future__ import annotations

import queue
import threading
from typing import Generic, TypeVar

T = TypeVar("T")


class DropOldestQueue(Generic[T]):
    """Cola FIFO acotada que descarta lo viejo en lugar de bloquear al productor."""

    def __init__(self, maxsize: int) -> None:
        """Args:
        maxsize: Capacidad máxima. Debe ser ≥ 1; valores pequeños (1-2) mantienen
            la latencia baja.
        """
        if maxsize < 1:
            raise ValueError("maxsize debe ser al menos 1.")
        self._queue: queue.Queue[T] = queue.Queue(maxsize=maxsize)
        self._lock = threading.Lock()
        self._dropped: int = 0

    @property
    def maxsize(self) -> int:
        """Capacidad máxima de la cola."""
        return self._queue.maxsize

    @property
    def dropped(self) -> int:
        """Cantidad total de elementos descartados desde la creación de la cola."""
        with self._lock:
            return self._dropped

    def put(self, item: T) -> bool:
        """Encola un elemento sin bloquear nunca.

        Returns:
            ``True`` si hubo que descartar un elemento viejo para hacer sitio.
        """
        with self._lock:
            descartado = False
            while True:
                try:
                    self._queue.put_nowait(item)
                    return descartado
                except queue.Full:
                    try:
                        self._queue.get_nowait()
                        self._dropped += 1
                        descartado = True
                    except queue.Empty:
                        # Otro consumidor vació la cola entre medio; reintentamos.
                        continue

    def get(self, timeout: float | None = None) -> T | None:
        """Extrae el elemento más antiguo disponible.

        Args:
            timeout: Segundos máximos de espera. ``None`` espera indefinidamente.

        Returns:
            El elemento, o ``None`` si se agotó el tiempo de espera.
        """
        try:
            if timeout is None:
                return self._queue.get()
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_latest(self, timeout: float | None = None) -> T | None:
        """Extrae el elemento más reciente y descarta los intermedios.

        Útil en la UI: dibujar frames viejos solo suma latencia percibida.
        """
        ultimo = self.get(timeout=timeout)
        if ultimo is None:
            return None
        while True:
            siguiente = self.get(timeout=0)
            if siguiente is None:
                return ultimo
            ultimo = siguiente

    def qsize(self) -> int:
        """Cantidad aproximada de elementos encolados."""
        return self._queue.qsize()

    def empty(self) -> bool:
        """``True`` si la cola está (aproximadamente) vacía."""
        return self._queue.empty()

    def clear(self) -> None:
        """Vacía la cola sin contabilizar descartes."""
        with self._lock:
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    return
