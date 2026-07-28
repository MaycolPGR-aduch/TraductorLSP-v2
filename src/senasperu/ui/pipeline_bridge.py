"""Puente entre los hilos de trabajo y la interfaz Qt.

Los hilos de captura y procesamiento son ``threading.Thread`` puros (se pueden
testear sin Qt ni GUI). Este objeto vive en el hilo de la UI, drena la cola de
resultados con un ``QTimer`` y reemite lo que encuentra como señales de Qt. Así
la UI solo conoce ``Signal``/``Slot`` y nunca toca OpenCV ni MediaPipe.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QTimer, Signal

from senasperu.capture.capture_thread import CaptureThread
from senasperu.capture.frame_queue import DropOldestQueue
from senasperu.capture.frame_source import Frame, FrameSource, create_frame_source
from senasperu.config import Config
from senasperu.features.holistic_thread import HolisticThread, ProcessedFrame
from senasperu.utils import FpsMeter

logger = logging.getLogger(__name__)

# Cada cuántos milisegundos se refrescan los contadores de rendimiento.
STATS_INTERVAL_MS: int = 500
# Cuántas veces por frame se consulta la cola de resultados (ver constructor).
POLL_OVERSAMPLING: int = 2

try:  # psutil es opcional: sin él, la app funciona pero sin métricas de sistema.
    import psutil

    _PROCESS = psutil.Process()
    _CPU_COUNT = psutil.cpu_count(logical=True) or 1
    _PROCESS.cpu_percent(None)  # primera llamada: inicializa el contador
except Exception:  # pragma: no cover - entorno sin psutil
    psutil = None  # type: ignore[assignment]
    _PROCESS = None
    _CPU_COUNT = 1


@dataclass(frozen=True, slots=True)
class PipelineStats:
    """Métricas de rendimiento del pipeline, para mostrar en pantalla."""

    capture_fps: float
    process_fps: float
    display_fps: float
    process_ms: float
    latency_ms: float
    frames_dropped: int
    hands_detected: int
    cpu_percent: float
    memory_mb: float
    elapsed_seconds: float


class PipelineBridge(QObject):
    """Arranca, detiene y expone a Qt el pipeline captura → MediaPipe."""

    frame_ready = Signal(object)  # ProcessedFrame
    stats_ready = Signal(object)  # PipelineStats
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
        video_path: Si se indica, se usa un archivo de video en vez de la webcam.
        parent: Padre Qt.
        """
        super().__init__(parent)
        self._config = config
        self._source: FrameSource = create_frame_source(config, video_path)

        cola_max = int(config.get("camara.cola_frames_max", 2))
        self._frame_queue: DropOldestQueue[Frame] = DropOldestQueue(cola_max)
        self._result_queue: DropOldestQueue[ProcessedFrame] = DropOldestQueue(cola_max)

        self._capture = CaptureThread(self._source, self._frame_queue)
        self._processing = HolisticThread(config, self._frame_queue, self._result_queue)

        fps_objetivo = int(config.get("camara.fps_objetivo", 30))
        self._poll_timer = QTimer(self)
        # PreciseTimer es obligatorio en Windows: el temporizador por defecto se
        # redondea a múltiplos de 15,6 ms, así que un intervalo de 33 ms termina
        # disparando cada 46,9 ms (21 FPS en vez de 30).
        self._poll_timer.setTimerType(Qt.TimerType.PreciseTimer)
        # Sondeamos al doble del ritmo de captura: así un frame recién llegado se
        # pinta enseguida en vez de esperar hasta un periodo completo. Cuando no
        # hay nada nuevo, la consulta a la cola es prácticamente gratis.
        self._poll_timer.setInterval(max(1, round(1000 / max(1, fps_objetivo * POLL_OVERSAMPLING))))
        self._poll_timer.timeout.connect(self._drain_results)

        self._stats_timer = QTimer(self)
        self._stats_timer.setInterval(STATS_INTERVAL_MS)
        self._stats_timer.timeout.connect(self._emit_stats)

        self._display_fps = FpsMeter()
        self._last_latency_ms = 0.0
        self._last_hands = 0
        self._started_at = 0.0
        self._running = False

    # -- Ciclo de vida -----------------------------------------------------
    def start(self) -> None:
        """Abre la fuente y arranca hilos y temporizadores.

        Raises:
            CameraError: Si la cámara o el archivo de video no se puede abrir.
        """
        if self._running:
            return
        self._capture.start()  # puede lanzar CameraError: lo maneja el llamador
        self._processing.start()
        self._started_at = time.perf_counter()
        self._poll_timer.start()
        self._stats_timer.start()
        self._running = True
        logger.info("Pipeline iniciado sobre %s", self._source.description)

    def stop(self) -> None:
        """Detiene temporizadores e hilos en orden y libera la cámara."""
        if not self._running:
            return
        self._running = False
        self._poll_timer.stop()
        self._stats_timer.stop()
        self._capture.stop()
        self._processing.stop()
        self._frame_queue.clear()
        self._result_queue.clear()
        logger.info("Pipeline detenido")

    @property
    def is_running(self) -> bool:
        """``True`` si el pipeline está activo."""
        return self._running

    def set_draw_landmarks(self, enabled: bool) -> None:
        """Activa o desactiva el dibujo del esqueleto en el hilo de procesamiento."""
        self._processing.draw_landmarks = enabled

    # -- Bombeo de la cola -------------------------------------------------
    def _drain_results(self) -> None:
        """Toma el resultado más reciente y lo publica como señal de Qt."""
        procesado = self._result_queue.get_latest(timeout=0)
        if procesado is not None:
            self._display_fps.tick()
            self._last_latency_ms = procesado.latency_seconds * 1000.0
            self._last_hands = procesado.result.hands_detected
            self.frame_ready.emit(procesado)

        error = self._capture.error or self._processing.error
        if error:
            self.stop()
            self.error_occurred.emit(error)

    def _emit_stats(self) -> None:
        """Publica las métricas de rendimiento acumuladas."""
        if _PROCESS is not None:
            cpu = _PROCESS.cpu_percent(None) / _CPU_COUNT
            memoria = _PROCESS.memory_info().rss / (1024 * 1024)
        else:  # pragma: no cover - entorno sin psutil
            cpu = float("nan")
            memoria = float("nan")

        self.stats_ready.emit(
            PipelineStats(
                capture_fps=self._capture.fps,
                process_fps=self._processing.fps,
                display_fps=self._display_fps.fps,
                process_ms=self._processing.process_ms,
                latency_ms=self._last_latency_ms,
                frames_dropped=self._capture.frames_dropped,
                hands_detected=self._last_hands,
                cpu_percent=cpu,
                memory_mb=memoria,
                elapsed_seconds=time.perf_counter() - self._started_at,
            )
        )
