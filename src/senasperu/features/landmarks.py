"""Estructuras y constantes de los landmarks entregados por MediaPipe Holistic.

Aquí vive el resultado *crudo* de un frame (todavía sin normalizar). La
normalización y el armado del vector de features son responsabilidad de la
Fase 1/2 y viven en otros módulos de este paquete.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Cantidades fijas de la solución Holistic.
POSE_LANDMARKS: int = 33  # cada uno con (x, y, z, visibility)
HAND_LANDMARKS: int = 21  # cada uno con (x, y, z)
FACE_LANDMARKS_FULL: int = 468  # FaceMesh sin refinamiento de iris

POSE_COORDS: int = 4
HAND_COORDS: int = 3
FACE_COORDS: int = 3

# Índices de pose usados como referencia de normalización (hombros).
POSE_LEFT_SHOULDER: int = 11
POSE_RIGHT_SHOULDER: int = 12

# Parte de la pose que importa en LSP: hombros, codos, muñecas y los puntos de
# mano que entrega el modelo de pose (11-22). Se excluyen los puntos faciales
# (0-10), que ya cubre FaceMesh, y caderas y piernas (23-32): cuando quedan
# fuera del encuadre MediaPipe los extrapola y solo ensucian la vista previa.
POSE_UPPER_BODY: frozenset[int] = frozenset(range(11, 23))


@dataclass(frozen=True, slots=True)
class HolisticResult:
    """Landmarks de un frame, en coordenadas normalizadas de imagen (0-1).

    Cualquier campo puede ser ``None`` si MediaPipe no detectó esa parte en el
    frame. Esa ausencia es información valiosa: alimenta el control de calidad
    del dataset y el manejo de huecos de detección.

    Attributes:
        pose: Matriz ``(33, 4)`` con ``x, y, z, visibility``.
        left_hand: Matriz ``(21, 3)`` de la mano izquierda.
        right_hand: Matriz ``(21, 3)`` de la mano derecha.
        face: Matriz ``(K, 3)`` con el subconjunto facial configurado.
        raw: Resultado original de MediaPipe (se usa solo para dibujar).
    """

    pose: np.ndarray | None = None
    left_hand: np.ndarray | None = None
    right_hand: np.ndarray | None = None
    face: np.ndarray | None = None
    raw: Any | None = None

    @property
    def hands_detected(self) -> int:
        """Cantidad de manos detectadas en el frame (0, 1 o 2)."""
        return int(self.left_hand is not None) + int(self.right_hand is not None)

    @property
    def has_pose(self) -> bool:
        """``True`` si se detectó el cuerpo."""
        return self.pose is not None

    @property
    def pose_visibility(self) -> float:
        """Visibilidad promedio de los landmarks de pose (0.0 si no hay pose).

        Sirve como señal de confianza por frame para el control de calidad.
        """
        if self.pose is None:
            return 0.0
        return float(np.mean(self.pose[:, 3]))

    def shoulders_visible(self, min_visibility: float) -> bool:
        """``True`` si ambos hombros superan la visibilidad mínima indicada.

        Los hombros son el punto de referencia y la escala de la normalización:
        sin ellos, el frame no es utilizable.

        Args:
            min_visibility: Umbral de visibilidad; sale de la configuración
                (``mediapipe.min_detection_confidence``).
        """
        if self.pose is None:
            return False
        return bool(
            self.pose[POSE_LEFT_SHOULDER, 3] >= min_visibility
            and self.pose[POSE_RIGHT_SHOULDER, 3] >= min_visibility
        )
