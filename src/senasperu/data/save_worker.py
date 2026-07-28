"""Hilo trabajador que escribe el dataset sin congelar la interfaz.

Guardar una repetición implica comprimir el ``.npz`` y codificar el ``.mp4`` de
respaldo: cientos de milisegundos. Hacerlo en el hilo de Qt cortaría la vista
previa justo entre una toma y la siguiente, así que se delega aquí.

La UI envía trabajos con :meth:`DatasetSaveWorker.submit` y recoge los
resultados con :meth:`poll` desde un ``QTimer``, igual que hace el puente del
pipeline de video.
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass

from senasperu.data.dataset_writer import DatasetWriter, SavedRecording
from senasperu.data.quality import QualityReport
from senasperu.data.recording import RecordingSample

logger = logging.getLogger(__name__)

# Espera del hilo por un trabajo nuevo antes de revisar si debe detenerse.
JOB_TIMEOUT_SECONDS: float = 0.2


@dataclass(frozen=True, slots=True)
class SaveResult:
    """Resultado de un trabajo de escritura.

    Attributes:
        saved: La grabación guardada, o ``None`` si el trabajo fue un descarte
            o si hubo error.
        discarded: Grabación que se eliminó, si el trabajo fue un descarte.
        error: Mensaje en español si algo falló.
    """

    saved: SavedRecording | None = None
    discarded: SavedRecording | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class _SaveJob:
    sample: RecordingSample
    person: str
    session: int
    report: QualityReport
    conditions: dict[str, str]


class DatasetSaveWorker(threading.Thread):
    """Cola de escritura del dataset en un hilo aparte."""

    def __init__(self, writer: DatasetWriter, *, name: str = "escritura-dataset") -> None:
        """Args:
        writer: Escritor del dataset (se usa solo desde este hilo).
        name: Nombre del hilo, aparece en los logs.
        """
        super().__init__(name=name, daemon=True)
        self._writer = writer
        self._jobs: queue.Queue[_SaveJob | SavedRecording | None] = queue.Queue()
        self._results: queue.Queue[SaveResult] = queue.Queue()
        self._stop_event = threading.Event()
        self._pending = 0
        self._pending_lock = threading.Lock()

    # -- API para la interfaz ---------------------------------------------
    def submit(
        self,
        sample: RecordingSample,
        *,
        person: str,
        session: int,
        report: QualityReport,
        conditions: dict[str, str],
    ) -> None:
        """Encola el guardado de una repetición."""
        self._increment()
        self._jobs.put(
            _SaveJob(
                sample=sample,
                person=person,
                session=session,
                report=report,
                conditions=dict(conditions),
            )
        )

    def submit_discard(self, saved: SavedRecording) -> None:
        """Encola el borrado de una repetición ya guardada."""
        self._increment()
        self._jobs.put(saved)

    def poll(self) -> SaveResult | None:
        """Devuelve el siguiente resultado disponible, o ``None`` si no hay."""
        try:
            return self._results.get_nowait()
        except queue.Empty:
            return None

    @property
    def pending(self) -> int:
        """Trabajos encolados o en curso."""
        with self._pending_lock:
            return self._pending

    # -- Ciclo de vida -----------------------------------------------------
    def run(self) -> None:  # noqa: D102 - documentado en la clase
        logger.info("Hilo de escritura del dataset iniciado")
        while not self._stop_event.is_set():
            try:
                trabajo = self._jobs.get(timeout=JOB_TIMEOUT_SECONDS)
            except queue.Empty:
                continue
            if trabajo is None:
                break
            self._results.put(self._process(trabajo))
            self._decrement()
        logger.info("Hilo de escritura del dataset detenido")

    def stop(self, timeout: float = 10.0) -> None:
        """Espera a que terminen los trabajos pendientes y detiene el hilo.

        No se descarta trabajo: perder una grabación recién hecha sería lo peor
        que podría pasar aquí.
        """
        self._jobs.put(None)
        if self.is_alive():
            self.join(timeout=timeout)
        self._stop_event.set()

    def _process(self, trabajo: _SaveJob | SavedRecording) -> SaveResult:
        try:
            if isinstance(trabajo, _SaveJob):
                guardada = self._writer.save(
                    trabajo.sample,
                    person=trabajo.person,
                    session=trabajo.session,
                    report=trabajo.report,
                    conditions=trabajo.conditions,
                )
                return SaveResult(saved=guardada)
            self._writer.discard(trabajo)
            return SaveResult(discarded=trabajo)
        except Exception as error:  # pragma: no cover - red de seguridad
            logger.exception("Error al escribir el dataset")
            return SaveResult(error=f"No se pudo escribir en el dataset: {error}")

    def _increment(self) -> None:
        with self._pending_lock:
            self._pending += 1

    def _decrement(self) -> None:
        with self._pending_lock:
            self._pending = max(0, self._pending - 1)
