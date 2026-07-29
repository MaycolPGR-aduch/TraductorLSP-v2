"""Puente entre el hilo de inferencia y la interfaz de traducción.

Mismo patrón que el puente del smoke test: los hilos de trabajo no conocen Qt, y
este objeto —que vive en el hilo de la interfaz— drena la cola de resultados con
un ``QTimer`` y los reemite como señales.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QTimer, Signal

from senasperu.capture.capture_thread import CaptureThread
from senasperu.capture.frame_queue import DropOldestQueue
from senasperu.capture.frame_source import Frame, FrameSource, create_frame_source
from senasperu.config import Config
from senasperu.features.translation_thread import TranslationFrame, TranslationThread
from senasperu.utils import FpsMeter

logger = logging.getLogger(__name__)

STATS_INTERVAL_MS: int = 500
POLL_OVERSAMPLING: int = 2


@dataclass(frozen=True, slots=True)
class TranslationStats:
    """Métricas del pipeline de traducción."""

    capture_fps: float
    process_fps: float
    display_fps: float
    process_ms: float
    inference_ms: float
    latency_ms: float
    frames_dropped: int


class TranslationBridge(QObject):
    """Arranca y detiene el pipeline de traducción y lo expone a Qt."""

    frame_ready = Signal(object)  # TranslationFrame
    stats_ready = Signal(object)  # TranslationStats
    error_occurred = Signal(str)

    def __init__(
        self,
        config: Config,
        *,
        video_path: str | Path | None = None,
        parent: QObject | None = None,
    ) -> None:
        """Args:
        config: Configuración cargada.
        video_path: Archivo de video en lugar de la webcam (pruebas).
        parent: Padre Qt.
        """
        super().__init__(parent)
        self._config = config
        self._source: FrameSource = create_frame_source(config, video_path)

        cola_max = int(config.get("camara.cola_frames_max", 2))
        self._frame_queue: DropOldestQueue[Frame] = DropOldestQueue(cola_max)
        self._result_queue: DropOldestQueue[TranslationFrame] = DropOldestQueue(cola_max)

        self._capture = CaptureThread(self._source, self._frame_queue)
        self._inference = TranslationThread(config, self._frame_queue, self._result_queue)

        fps_objetivo = int(config.get("camara.fps_objetivo", 30))
        self._poll_timer = QTimer(self)
        self._poll_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._poll_timer.setInterval(max(1, round(1000 / max(1, fps_objetivo * POLL_OVERSAMPLING))))
        self._poll_timer.timeout.connect(self._drain_results)

        self._stats_timer = QTimer(self)
        self._stats_timer.setInterval(STATS_INTERVAL_MS)
        self._stats_timer.timeout.connect(self._emit_stats)

        self._display_fps = FpsMeter()
        self._last_latency_ms = 0.0
        self._running = False

    def start(self) -> None:
        """Abre la cámara y arranca hilos y temporizadores."""
        if self._running:
            return
        self._capture.start()
        self._inference.start()
        self._poll_timer.start()
        self._stats_timer.start()
        self._running = True
        logger.info("Pipeline de traducción iniciado sobre %s", self._source.description)

    def stop(self) -> None:
        """Detiene todo en orden y libera la cámara."""
        if not self._running:
            return
        self._running = False
        self._poll_timer.stop()
        self._stats_timer.stop()
        self._capture.stop()
        self._inference.stop()
        self._frame_queue.clear()
        self._result_queue.clear()
        logger.info("Pipeline de traducción detenido")

    @property
    def is_running(self) -> bool:
        """``True`` si el pipeline está activo."""
        return self._running

    def set_draw_landmarks(self, enabled: bool) -> None:
        """Activa o desactiva el dibujo del esqueleto."""
        self._inference.draw_landmarks = enabled

    def _drain_results(self) -> None:
        """Toma el resultado más reciente y lo publica como señal."""
        resultado = self._result_queue.get_latest(timeout=0)
        if resultado is not None:
            self._display_fps.tick()
            self._last_latency_ms = resultado.latency_seconds * 1000.0
            self.frame_ready.emit(resultado)

        error = self._capture.error or self._inference.error
        if error:
            self.stop()
            self.error_occurred.emit(error)

    def _emit_stats(self) -> None:
        """Publica las métricas de rendimiento."""
        self.stats_ready.emit(
            TranslationStats(
                capture_fps=self._capture.fps,
                process_fps=self._inference.fps,
                display_fps=self._display_fps.fps,
                process_ms=self._inference.process_ms,
                inference_ms=self._inference.inference_ms,
                latency_ms=self._last_latency_ms,
                frames_dropped=self._capture.frames_dropped,
            )
        )
