"""Envoltorio de MediaPipe Holistic.

Aísla toda la dependencia de MediaPipe en un solo lugar: el resto del proyecto
solo conoce :class:`~senasperu.features.landmarks.HolisticResult`. Si alguna vez
hay que migrar a la API Tasks (``HolisticLandmarker``), este es el único módulo
que cambia... pero el dataset ya grabado habría que re-extraerlo desde los videos
de respaldo, porque los landmarks no serían idénticos.

Requiere ``mediapipe==0.10.21``: es la última versión que incluye las soluciones
legacy con sus modelos ``.tflite`` empaquetados dentro de la wheel.

Nota de rendimiento: la solución Holistic *siempre* ejecuta el modelo facial, no
se puede desactivar. Por eso ``mediapipe.usar_rostro: false`` en la configuración
significa "no uses el rostro como feature ni lo dibujes", no "no lo calcules".
Si en el benchmark de Fase 0 el rostro resulta demasiado caro para el equipo
objetivo, la palanca real es ``model_complexity: 0``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Sequence

import numpy as np

from senasperu.config import Config
from senasperu.features.landmarks import (
    FACE_LANDMARKS_FULL,
    HAND_LANDMARKS,
    POSE_LANDMARKS,
    HolisticResult,
)

logger = logging.getLogger(__name__)

MENSAJE_SIN_MEDIAPIPE = (
    "No se pudo cargar MediaPipe Holistic. Instala las dependencias con "
    "'pip install -e .' dentro del entorno virtual del proyecto."
)


class HolisticExtractor:
    """Ejecuta MediaPipe Holistic sobre frames BGR y devuelve landmarks.

    No es seguro compartir una instancia entre hilos: cada hilo debe crear la
    suya (MediaPipe mantiene estado temporal de tracking).
    """

    def __init__(
        self,
        *,
        model_complexity: int,
        min_detection_confidence: float,
        min_tracking_confidence: float,
        use_face: bool,
        face_indices: Sequence[int] | None,
    ) -> None:
        """Args:
        model_complexity: 0 (rápido), 1 (balance) o 2 (preciso).
        min_detection_confidence: Confianza mínima para detectar.
        min_tracking_confidence: Confianza mínima para seguir entre frames.
        use_face: Si se extraen (y dibujan) landmarks faciales.
        face_indices: Subconjunto de índices de FaceMesh a conservar. Si es
            ``None`` y ``use_face`` es ``True``, se conserva el rostro completo.
        """
        self._use_face = use_face
        self._face_indices = np.asarray(face_indices, dtype=np.int32) if face_indices else None
        self._closed = False

        # MediaPipe vuelca avisos de su capa C++ en stderr y ensucian los logs
        # del usuario. Hay que silenciarlos ANTES de importarlo.
        os.environ.setdefault("GLOG_minloglevel", "2")
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

        try:
            import mediapipe as mp  # import diferido: pesa y solo hace falta aquí
        except ImportError as error:  # pragma: no cover - depende del entorno
            raise RuntimeError(MENSAJE_SIN_MEDIAPIPE) from error

        solucion = getattr(getattr(mp, "solutions", None), "holistic", None)
        if solucion is None:  # pragma: no cover - versión equivocada instalada
            raise RuntimeError(
                f"La versión instalada de MediaPipe ({getattr(mp, '__version__', '?')}) ya no "
                "incluye la solución Holistic: Google la eliminó a partir de la 0.10.30. "
                "Reinstala la versión fijada del proyecto con 'pip install -e .' "
                "(mediapipe==0.10.21)."
            )

        self._holistic = solucion.Holistic(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,      # no lo usamos: puro costo de CPU
            refine_face_landmarks=False,    # el iris no aporta a la LSP
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        logger.info(
            "MediaPipe Holistic listo (complejidad=%s, rostro=%s)",
            model_complexity,
            "sí" if use_face else "no",
        )

    @classmethod
    def from_config(cls, config: Config) -> HolisticExtractor:
        """Construye el extractor con la sección ``mediapipe`` del YAML."""
        usar_rostro = bool(config.get("mediapipe.usar_rostro", True))
        rostro_reducido = bool(config.get("mediapipe.rostro_reducido", True))
        indices = config.get("mediapipe.indices_rostro") if rostro_reducido else None
        return cls(
            model_complexity=int(config.require("mediapipe.model_complexity")),
            min_detection_confidence=float(config.require("mediapipe.min_detection_confidence")),
            min_tracking_confidence=float(config.require("mediapipe.min_tracking_confidence")),
            use_face=usar_rostro,
            face_indices=indices,
        )

    def process(self, image_bgr: np.ndarray) -> HolisticResult:
        """Procesa un frame BGR y devuelve sus landmarks.

        Args:
            image_bgr: Imagen en formato BGR (el que entrega OpenCV).

        Returns:
            Los landmarks del frame; los campos no detectados quedan en ``None``.
        """
        if self._closed:
            raise RuntimeError("El extractor de MediaPipe ya fue cerrado.")

        import cv2  # local: mantiene el módulo importable sin OpenCV en tests unitarios

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        # Marcar la imagen como no escribible evita una copia interna de MediaPipe.
        image_rgb.flags.writeable = False
        resultado = self._holistic.process(image_rgb)

        return HolisticResult(
            pose=_to_array(getattr(resultado, "pose_landmarks", None), POSE_LANDMARKS, True),
            left_hand=_to_array(
                getattr(resultado, "left_hand_landmarks", None), HAND_LANDMARKS, False
            ),
            right_hand=_to_array(
                getattr(resultado, "right_hand_landmarks", None), HAND_LANDMARKS, False
            ),
            face=self._face_array(getattr(resultado, "face_landmarks", None)),
            raw=resultado,
        )

    def close(self) -> None:
        """Libera los recursos de MediaPipe. Es idempotente."""
        if not self._closed:
            self._holistic.close()
            self._closed = True

    def __enter__(self) -> HolisticExtractor:  # noqa: D105
        return self

    def __exit__(self, *_exc_info) -> None:  # noqa: D105
        self.close()

    def _face_array(self, landmarks: Any | None) -> np.ndarray | None:
        if not self._use_face or landmarks is None:
            return None
        arreglo = _to_array(landmarks, FACE_LANDMARKS_FULL, False)
        if arreglo is None or self._face_indices is None:
            return arreglo
        if int(self._face_indices.max(initial=-1)) >= arreglo.shape[0]:
            logger.warning(
                "Los índices faciales de la configuración exceden los %s puntos "
                "entregados por MediaPipe; se usa el rostro completo.",
                arreglo.shape[0],
            )
            return arreglo
        return arreglo[self._face_indices]


def _to_array(landmarks: Any | None, expected: int, with_visibility: bool) -> np.ndarray | None:
    """Convierte una lista de landmarks de MediaPipe en un ``np.ndarray`` float32."""
    if landmarks is None:
        return None
    puntos = getattr(landmarks, "landmark", None)
    if not puntos:
        return None
    if with_visibility:
        datos = [(p.x, p.y, p.z, p.visibility) for p in puntos]
    else:
        datos = [(p.x, p.y, p.z) for p in puntos]
    arreglo = np.asarray(datos, dtype=np.float32)
    if arreglo.shape[0] != expected:
        logger.debug(
            "MediaPipe devolvió %s landmarks donde se esperaban %s", arreglo.shape[0], expected
        )
    return arreglo
