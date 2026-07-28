"""Hilo dedicado de captura de video.

Lee de una :class:`~senasperu.capture.frame_source.FrameSource` tan rápido como el
dispositivo entregue frames y los publica en una
:class:`~senasperu.capture.frame_queue.DropOldestQueue`. Nunca bloquea: si el
consumidor va lento, el frame viejo se descarta.
"""

from __future__ import annotations

import logging
import threading
import time

from senasperu.capture.frame_queue import DropOldestQueue
from senasperu.capture.frame_source import CameraError, Frame, FrameSource
from senasperu.utils import FpsMeter

logger = logging.getLogger(__name__)

# Lecturas fallidas seguidas antes de dar la cámara por perdida.
MAX_CONSECUTIVE_FAILURES: int = 30
# Pausa breve tras una lectura fallida, para no quemar CPU reintentando.
RETRY_SLEEP_SECONDS: float = 0.01


class CaptureThread(threading.Thread):
    """Hilo productor de frames."""

    def __init__(
        self,
        source: FrameSource,
        frame_queue: DropOldestQueue[Frame],
        *,
        name: str = "captura",
    ) -> None:
        """Args:
        source: Fuente de frames (webcam o archivo de video).
        frame_queue: Cola de salida hacia el hilo de procesamiento.
        name: Nombre del hilo (aparece en los logs).
        """
        super().__init__(name=name, daemon=True)
        self._source = source
        self._queue = frame_queue
        self._stop_event = threading.Event()
        self._fps_meter = FpsMeter()
        self._error: str | None = None
        self._frames_read = 0

    # -- Ciclo de vida -----------------------------------------------------
    def start(self) -> None:
        """Abre la fuente y arranca el hilo.

        Raises:
            CameraError: Si la fuente no se puede abrir (se propaga al llamador
                para poder mostrar un mensaje al usuario antes de abrir la UI).
        """
        self._source.open()
        super().start()

    def run(self) -> None:  # noqa: D102 - documentado en la clase
        logger.info("Hilo de captura iniciado (%s)", self._source.description)
        fallos = 0
        try:
            while not self._stop_event.is_set():
                frame = self._source.read()
                if frame is None:
                    fallos += 1
                    if fallos >= MAX_CONSECUTIVE_FAILURES:
                        self._error = (
                            "Se perdió la señal de la cámara. Revisa la conexión "
                            "y vuelve a iniciar la aplicación."
                        )
                        logger.error(self._error)
                        break
                    time.sleep(RETRY_SLEEP_SECONDS)
                    continue

                fallos = 0
                self._frames_read += 1
                self._fps_meter.tick(frame.timestamp)
                self._queue.put(frame)
        except CameraError as error:  # pragma: no cover - depende del hardware
            self._error = str(error)
            logger.exception("Error de cámara en el hilo de captura")
        except Exception as error:  # pragma: no cover - red de seguridad
            self._error = f"Error inesperado en la captura: {error}"
            logger.exception("Error inesperado en el hilo de captura")
        finally:
            self._source.close()
            logger.info(
                "Hilo de captura detenido: %s frames leídos, %s descartados",
                self._frames_read,
                self._queue.dropped,
            )

    def stop(self, timeout: float = 2.0) -> None:
        """Pide la detención del hilo y espera a que termine."""
        self._stop_event.set()
        if self.is_alive():
            self.join(timeout=timeout)

    # -- Estado ------------------------------------------------------------
    @property
    def fps(self) -> float:
        """FPS de captura medidos por media móvil."""
        return self._fps_meter.fps

    @property
    def frames_read(self) -> int:
        """Total de frames leídos con éxito."""
        return self._frames_read

    @property
    def frames_dropped(self) -> int:
        """Total de frames descartados por cola llena."""
        return self._queue.dropped

    @property
    def error(self) -> str | None:
        """Mensaje de error en español si el hilo murió por un fallo; si no, ``None``."""
        return self._error
