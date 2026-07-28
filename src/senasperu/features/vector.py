"""Serialización de landmarks a un vector plano de features.

Los vectores que se guardan en el dataset son **crudos**: coordenadas tal como
las entrega MediaPipe, sin normalizar. La normalización se aplica al cargar
(Fase 2), de modo que se pueda cambiar el criterio de normalización sin tener
que regrabar una sola repetición.

Las partes no detectadas se rellenan con ``NaN``, no con ceros: un cero es una
coordenada válida (esquina superior izquierda) y confundiría "mano en el borde"
con "mano ausente". El manejo de huecos de Fase 2 depende de esa distinción.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from senasperu.config import Config
from senasperu.features.landmarks import (
    HAND_COORDS,
    HAND_LANDMARKS,
    POSE_COORDS,
    POSE_LANDMARKS,
    HolisticResult,
)

# Valor con que se marcan los landmarks no detectados.
MISSING: float = float("nan")

BLOCK_POSE: str = "pose"
BLOCK_LEFT_HAND: str = "left_hand"
BLOCK_RIGHT_HAND: str = "right_hand"
BLOCK_FACE: str = "face"


@dataclass(frozen=True, slots=True)
class FeatureBlock:
    """Un tramo del vector de features.

    Attributes:
        name: Nombre del bloque (``pose``, ``left_hand``, ...).
        start: Índice donde empieza dentro del vector.
        points: Cantidad de landmarks del bloque.
        coords: Valores por landmark (3 = x,y,z; 4 = x,y,z,visibility).
    """

    name: str
    start: int
    points: int
    coords: int

    @property
    def size(self) -> int:
        """Cantidad de valores que ocupa el bloque."""
        return self.points * self.coords

    @property
    def slice(self) -> slice:
        """Rebanada del vector correspondiente a este bloque."""
        return slice(self.start, self.start + self.size)


@dataclass(frozen=True, slots=True)
class FeatureLayout:
    """Descripción del vector de features: qué hay y en qué orden.

    Se guarda dentro de cada ``.npz`` para que los archivos sean
    autodescriptivos: si mañana cambia el subconjunto facial de la
    configuración, los datos viejos siguen siendo interpretables.
    """

    blocks: tuple[FeatureBlock, ...]

    @property
    def size(self) -> int:
        """Largo total del vector de features."""
        return sum(bloque.size for bloque in self.blocks)

    @property
    def names(self) -> tuple[str, ...]:
        """Nombres de los bloques, en orden."""
        return tuple(bloque.name for bloque in self.blocks)

    def block(self, name: str) -> FeatureBlock:
        """Devuelve el bloque con ese nombre.

        Raises:
            KeyError: Si el bloque no forma parte de este layout.
        """
        for bloque in self.blocks:
            if bloque.name == name:
                return bloque
        raise KeyError(f"El vector de features no tiene el bloque '{name}'.")

    def has(self, name: str) -> bool:
        """``True`` si el layout incluye ese bloque."""
        return any(bloque.name == name for bloque in self.blocks)


def build_layout(*, face_points: int) -> FeatureLayout:
    """Construye el layout del vector de features.

    Args:
        face_points: Cantidad de puntos faciales que se conservan. Si es 0, el
            vector no incluye bloque facial.
    """
    bloques: list[FeatureBlock] = []
    inicio = 0
    for nombre, puntos, coords in (
        (BLOCK_POSE, POSE_LANDMARKS, POSE_COORDS),
        (BLOCK_LEFT_HAND, HAND_LANDMARKS, HAND_COORDS),
        (BLOCK_RIGHT_HAND, HAND_LANDMARKS, HAND_COORDS),
        (BLOCK_FACE, face_points, 3),
    ):
        if puntos <= 0:
            continue
        bloque = FeatureBlock(name=nombre, start=inicio, points=puntos, coords=coords)
        bloques.append(bloque)
        inicio += bloque.size
    return FeatureLayout(blocks=tuple(bloques))


def layout_from_config(config: Config) -> FeatureLayout:
    """Construye el layout según la sección ``mediapipe`` de la configuración."""
    if not bool(config.get("mediapipe.usar_rostro", True)):
        return build_layout(face_points=0)
    if not bool(config.get("mediapipe.rostro_reducido", True)):
        from senasperu.features.landmarks import FACE_LANDMARKS_FULL

        return build_layout(face_points=FACE_LANDMARKS_FULL)
    indices = config.get("mediapipe.indices_rostro") or []
    return build_layout(face_points=len(indices))


def to_feature_vector(result: HolisticResult, layout: FeatureLayout) -> np.ndarray:
    """Aplana un resultado de MediaPipe al vector de features del layout.

    Args:
        result: Landmarks del frame.
        layout: Layout que define el orden y el tamaño del vector.

    Returns:
        Vector ``float32`` de largo ``layout.size``; los bloques no detectados
        quedan llenos de ``NaN``.
    """
    vector = np.full(layout.size, MISSING, dtype=np.float32)
    fuentes = {
        BLOCK_POSE: result.pose,
        BLOCK_LEFT_HAND: result.left_hand,
        BLOCK_RIGHT_HAND: result.right_hand,
        BLOCK_FACE: result.face,
    }
    for bloque in layout.blocks:
        datos = fuentes.get(bloque.name)
        if datos is None:
            continue
        if datos.shape != (bloque.points, bloque.coords):
            # Tamaño inesperado: es preferible dejar el bloque como ausente
            # antes que escribir datos desalineados en el dataset.
            continue
        vector[bloque.slice] = datos.reshape(-1)
    return vector
