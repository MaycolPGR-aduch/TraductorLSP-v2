"""Hilo de procesamiento: frames → MediaPipe Holistic → landmarks (+ overlay).

En Fase 3 este mismo hilo sumará normalización, ventana deslizante, modelo ONNX
y estabilización. En Fase 0 solo extrae landmarks y los dibuja, que es lo que el
smoke test necesita medir.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import numpy as np

from senasperu.capture.frame_queue import DropOldestQueue
from senasperu.capture.frame_source import Frame
from senasperu.config import Config
from senasperu.features.holistic import HolisticExtractor
from senasperu.features.landmarks import HolisticResult
from senasperu.features.overlay import LandmarkOverlay
from senasperu.utils import FpsMeter, LatencyMeter

logger = logging.getLogger(__name__)

# Espera máxima por un frame nuevo antes de revisar si hay que detenerse.
QUEUE_TIMEOUT_SECONDS: float = 0.2


@dataclass(frozen=True, slots=True)
class ProcessedFrame:
    """Resultado listo para mostrar en pantalla.

    Attributes:
        image: Imagen BGR, ya con el esqueleto dibujado si estaba habilitado.
        result: Landmarks extraídos del frame.
        frame_index: Índice del frame de origen.
        capture_timestamp: Instante en que se capturó el frame.
        process_seconds: Tiempo que tomó MediaPipe en este frame.
        latency_seconds: Tiempo total desde la captura hasta el fin del proceso.
    """

    image: np.ndarray
    result: HolisticResult
    frame_index: int
    capture_timestamp: float
    process_seconds: float
    latency_seconds: float


class HolisticThread(threading.Thread):
    """Hilo consumidor de frames y productor de resultados."""

    def __init__(
        self,
        config: Config,
        frame_queue: DropOldestQueue[Frame],
        result_queue: DropOldestQueue[ProcessedFrame],
        *,
        draw_landmarks: bool | None = None,
        name: str = "procesamiento",
    ) -> None:
        """Args:
        config: Configuración cargada.
        frame_queue: Cola de entrada alimentada por el hilo de captura.
        result_queue: Cola de salida hacia la UI.
        draw_landmarks: Si se dibuja el esqueleto. Si es ``None``, se toma de
            ``ui.mostrar_landmarks``.
        name: Nombre del hilo.
        """
        super().__init__(name=name, daemon=True)
        self._config = config
        self._frame_queue = frame_queue
        self._result_queue = result_queue
        self._stop_event = threading.Event()
        self._fps_meter = FpsMeter()
        self._latency_meter = LatencyMeter()
        self._error: str | None = None
        self._frames_processed = 0
        self.draw_landmarks: bool = (
            bool(config.get("ui.mostrar_landmarks", True))
            if draw_landmarks is None
            else draw_landmarks
        )

    def run(self) -> None:  # noqa: D102 - documentado en la clase
        logger.info("Hilo de procesamiento iniciado")
        extractor: HolisticExtractor | None = None
        try:
            # MediaPipe mantiene estado de tracking: la instancia se crea y se usa
            # siempre en este hilo.
            extractor = HolisticExtractor.from_config(self._config)
            overlay = LandmarkOverlay(draw_face=bool(self._config.get("mediapipe.usar_rostro", True)))

            while not self._stop_event.is_set():
                frame = self._frame_queue.get(timeout=QUEUE_TIMEOUT_SECONDS)
                if frame is None:
                    continue

                inicio = time.perf_counter()
                resultado = extractor.process(frame.image)
                fin_proceso = time.perf_counter()

                imagen = frame.image
                if self.draw_landmarks:
                    imagen = overlay.draw(imagen, resultado)

                ahora = time.perf_counter()
                self._frames_processed += 1
                self._fps_meter.tick(ahora)
                self._latency_meter.add(fin_proceso - inicio)

                self._result_queue.put(
                    ProcessedFrame(
                        image=imagen,
                        result=resultado,
                        frame_index=frame.index,
                        capture_timestamp=frame.timestamp,
                        process_seconds=fin_proceso - inicio,
                        latency_seconds=ahora - frame.timestamp,
                    )
                )
        except Exception as error:  # pragma: no cover - red de seguridad
            self._error = f"Error en el procesamiento de video: {error}"
            logger.exception("Error en el hilo de procesamiento")
        finally:
            if extractor is not None:
                extractor.close()
            logger.info(
                "Hilo de procesamiento detenido: %s frames procesados", self._frames_processed
            )

    def stop(self, timeout: float = 3.0) -> None:
        """Pide la detención del hilo y espera a que termine."""
        self._stop_event.set()
        if self.is_alive():
            self.join(timeout=timeout)

    # -- Estado ------------------------------------------------------------
    @property
    def fps(self) -> float:
        """FPS de procesamiento (frames que realmente pasan por MediaPipe)."""
        return self._fps_meter.fps

    @property
    def process_ms(self) -> float:
        """Tiempo promedio de MediaPipe por frame, en milisegundos."""
        return self._latency_meter.milliseconds

    @property
    def frames_processed(self) -> int:
        """Total de frames procesados."""
        return self._frames_processed

    @property
    def error(self) -> str | None:
        """Mensaje de error en español si el hilo murió por un fallo."""
        return self._error
