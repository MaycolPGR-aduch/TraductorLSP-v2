"""Acumulación de una repetición mientras se graba.

El buffer recibe los landmarks frame a frame y, al cerrar, entrega una
:class:`RecordingSample` lista para evaluar su calidad y guardarla.

No depende de OpenCV ni de Qt: recibe los datos ya extraídos, así que se puede
testear con landmarks sintéticos.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from senasperu.features.landmarks import HolisticResult
from senasperu.features.vector import FeatureLayout, to_feature_vector


@dataclass(frozen=True, slots=True)
class RecordingSample:
    """Una repetición completa, lista para guardarse.

    Attributes:
        label: Id de la seña (``hola``, ``gracias``, ``no_sena``...).
        landmarks: Matriz ``(frames, features)`` de landmarks crudos, con ``NaN``
            donde no hubo detección.
        confidence: Confianza por frame.
        hands_per_frame: Manos detectadas en cada frame (0, 1 o 2).
        fps: FPS reales medidos durante la grabación.
        layout: Descripción del vector de features.
        video_frames: Frames BGR sin overlay, si se pidió respaldo de video.
    """

    label: str
    landmarks: np.ndarray
    confidence: np.ndarray
    hands_per_frame: np.ndarray
    fps: float
    layout: FeatureLayout
    video_frames: tuple[np.ndarray, ...] | None = None

    @property
    def frames(self) -> int:
        """Cantidad de frames de la grabación."""
        return int(self.landmarks.shape[0])

    @property
    def duration_seconds(self) -> float:
        """Duración de la grabación según los FPS medidos."""
        return self.frames / self.fps if self.fps > 0 else 0.0


class RecordingBuffer:
    """Acumula los frames de una repetición en curso."""

    def __init__(self, layout: FeatureLayout, *, keep_video: bool = False) -> None:
        """Args:
        layout: Layout del vector de features.
        keep_video: Si se conservan los frames BGR para el video de respaldo.
        """
        self._layout = layout
        self._keep_video = keep_video
        self._label: str = ""
        self._vectors: list[np.ndarray] = []
        self._confidence: list[float] = []
        self._hands: list[int] = []
        self._timestamps: list[float] = []
        self._images: list[np.ndarray] = []

    def start(self, label: str) -> None:
        """Descarta lo acumulado y comienza a grabar la seña indicada."""
        self._label = label
        self._vectors.clear()
        self._confidence.clear()
        self._hands.clear()
        self._timestamps.clear()
        self._images.clear()

    def add(
        self,
        result: HolisticResult,
        timestamp: float,
        image: np.ndarray | None = None,
    ) -> None:
        """Agrega un frame a la grabación en curso.

        Args:
            result: Landmarks del frame.
            timestamp: Marca de tiempo monótona de la captura.
            image: Frame BGR **sin overlay**, necesario si se guarda respaldo.
        """
        self._vectors.append(to_feature_vector(result, self._layout))
        self._confidence.append(result.pose_visibility)
        self._hands.append(result.hands_detected)
        self._timestamps.append(timestamp)
        if self._keep_video and image is not None:
            # Copia obligatoria: el frame original se reutiliza aguas arriba.
            self._images.append(image.copy())

    def build(self) -> RecordingSample:
        """Cierra la grabación y devuelve la muestra acumulada."""
        if self._vectors:
            landmarks = np.stack(self._vectors).astype(np.float32)
        else:
            landmarks = np.empty((0, self._layout.size), dtype=np.float32)
        return RecordingSample(
            label=self._label,
            landmarks=landmarks,
            confidence=np.asarray(self._confidence, dtype=np.float32),
            hands_per_frame=np.asarray(self._hands, dtype=np.int8),
            fps=self.measured_fps,
            layout=self._layout,
            video_frames=tuple(self._images) if self._keep_video else None,
        )

    @property
    def frames(self) -> int:
        """Frames acumulados hasta ahora."""
        return len(self._vectors)

    @property
    def elapsed_seconds(self) -> float:
        """Segundos transcurridos desde el primer frame acumulado."""
        if len(self._timestamps) < 2:
            return 0.0
        return self._timestamps[-1] - self._timestamps[0]

    @property
    def measured_fps(self) -> float:
        """FPS reales de la grabación, calculados con las marcas de tiempo."""
        transcurrido = self.elapsed_seconds
        if transcurrido <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / transcurrido
