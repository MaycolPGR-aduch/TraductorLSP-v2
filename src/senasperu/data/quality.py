"""Control de calidad de las grabaciones del dataset.

Se evalúa **en el momento**, justo al terminar cada repetición: es mucho más
barato regrabar en el acto que descubrir meses después que media sesión no sirve.

Lógica pura: no depende de cámara, Qt ni MediaPipe, y se testea con arreglos.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from senasperu.config import Config


@dataclass(frozen=True, slots=True)
class QualityReport:
    """Veredicto de calidad de una repetición grabada.

    Attributes:
        accepted: ``True`` si la grabación se puede guardar.
        frames: Cantidad de frames capturados.
        frames_without_hands: Frames en los que no se detectó ninguna mano.
        without_hands_pct: Porcentaje de frames sin manos.
        mean_confidence: Confianza promedio por frame.
        reasons: Motivos del rechazo, en español y listos para mostrar.
    """

    accepted: bool
    frames: int
    frames_without_hands: int
    without_hands_pct: float
    mean_confidence: float
    reasons: tuple[str, ...] = field(default=())

    @property
    def summary(self) -> str:
        """Texto corto para la interfaz y el log."""
        if self.accepted:
            return (
                f"Aceptada — {self.frames} frames, "
                f"{self.without_hands_pct:.0f} % sin manos, "
                f"confianza {self.mean_confidence:.2f}"
            )
        return "Rechazada — " + " ".join(self.reasons)


class QualityChecker:
    """Aplica los umbrales de ``calidad_datos`` a una grabación."""

    def __init__(
        self,
        *,
        max_frames_without_hands_pct: float,
        min_mean_confidence: float,
        min_frames: int,
        auto_reject: bool = True,
    ) -> None:
        """Args:
        max_frames_without_hands_pct: Máximo de frames sin manos tolerado (0-100).
        min_mean_confidence: Confianza promedio mínima aceptable.
        min_frames: Mínimo de frames para considerar la grabación completa.
        auto_reject: Si es ``False``, se calcula el informe pero nada se rechaza
            (útil para inspeccionar sin bloquear la grabación).
        """
        self._max_without_hands_pct = max_frames_without_hands_pct
        self._min_mean_confidence = min_mean_confidence
        self._min_frames = min_frames
        self._auto_reject = auto_reject

    @classmethod
    def from_config(cls, config: Config) -> QualityChecker:
        """Construye el verificador con los umbrales del YAML.

        El mínimo de frames se deriva de la duración de grabación y los FPS
        objetivo: se exige la mitad de lo esperado para dar margen a cámaras
        que no alcanzan su FPS nominal.
        """
        fps = float(config.require("camara.fps_objetivo"))
        duracion = float(config.require("grabador.duracion_grabacion_segundos"))
        return cls(
            max_frames_without_hands_pct=float(
                config.require("calidad_datos.max_frames_sin_manos_pct")
            ),
            min_mean_confidence=float(config.require("calidad_datos.confianza_minima_promedio")),
            min_frames=max(1, int(fps * duracion * 0.5)),
            auto_reject=bool(config.get("calidad_datos.rechazo_automatico", True)),
        )

    def evaluate(
        self,
        hands_per_frame: Sequence[int] | np.ndarray,
        confidence: Sequence[float] | np.ndarray,
    ) -> QualityReport:
        """Evalúa una grabación.

        Args:
            hands_per_frame: Manos detectadas en cada frame (0, 1 o 2).
            confidence: Confianza por frame (visibilidad promedio de la pose).

        Returns:
            El informe con el veredicto y los motivos.
        """
        manos = np.asarray(hands_per_frame, dtype=np.int16)
        confianzas = np.asarray(confidence, dtype=np.float32)
        total = int(manos.size)

        if total == 0:
            return QualityReport(
                accepted=False,
                frames=0,
                frames_without_hands=0,
                without_hands_pct=100.0,
                mean_confidence=0.0,
                reasons=("No se capturó ningún frame. Revisa que la cámara esté funcionando.",),
            )

        sin_manos = int(np.count_nonzero(manos <= 0))
        pct_sin_manos = 100.0 * sin_manos / total
        confianza_media = float(np.mean(confianzas)) if confianzas.size else 0.0

        motivos: list[str] = []
        if total < self._min_frames:
            motivos.append(
                f"Solo se capturaron {total} frames (mínimo {self._min_frames}). "
                "La cámara puede estar entregando menos FPS de los esperados."
            )
        if pct_sin_manos > self._max_without_hands_pct:
            motivos.append(
                f"Se perdieron las manos en {pct_sin_manos:.0f} % de los frames "
                f"(máximo {self._max_without_hands_pct:.0f} %). "
                "Mantén las manos dentro del encuadre y mejora la iluminación."
            )
        if confianza_media < self._min_mean_confidence:
            motivos.append(
                f"Confianza promedio {confianza_media:.2f}, por debajo del mínimo "
                f"{self._min_mean_confidence:.2f}. Aléjate un poco para que se vea el torso."
            )

        aceptada = not motivos or not self._auto_reject
        return QualityReport(
            accepted=aceptada,
            frames=total,
            frames_without_hands=sin_manos,
            without_hands_pct=pct_sin_manos,
            mean_confidence=confianza_media,
            reasons=tuple(motivos),
        )
