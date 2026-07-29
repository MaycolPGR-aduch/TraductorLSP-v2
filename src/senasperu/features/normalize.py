"""Normalización de landmarks: de coordenadas de imagen a un espacio comparable.

Sin esto el modelo aprende la posición de la persona frente a la cámara en lugar
de la seña. Tres correcciones, en este orden:

1. **Isotropía.** MediaPipe entrega ``x`` e ``y`` normalizados al ancho y al alto
   de la imagen. En un cuadro 4:3 eso significa que un desplazamiento horizontal
   y uno vertical del mismo tamaño real dan números distintos. Se corrige
   multiplicando ``x`` por la relación de aspecto.
2. **Traslación.** Se centra todo en el punto medio de los hombros, que es el
   punto estable del torso.
3. **Escala.** Se divide por la distancia entre hombros, lo que da invarianza a
   la distancia de la cámara y al tamaño del cuerpo.

Además se rellenan los huecos cortos de detección (≤3 frames) por interpolación
lineal, y a cada parte del cuerpo se le añade un indicador de presencia: que una
mano **no** esté es información útil (hay señas de una sola mano), y rellenar con
ceros la confundiría con "mano en el centro del pecho".

Es lógica pura: sin cámara, sin MediaPipe y sin PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from senasperu.config import Config
from senasperu.features.landmarks import POSE_LEFT_SHOULDER, POSE_RIGHT_SHOULDER
from senasperu.features.vector import BLOCK_POSE, FeatureLayout

# Distancia entre hombros por debajo de la cual el frame se considera inservible
# (la persona está de perfil o la detección colapsó).
MIN_SHOULDER_DISTANCE: float = 1e-3


@dataclass(frozen=True, slots=True)
class NormalizedSequence:
    """Secuencia normalizada y lista para el modelo.

    Attributes:
        features: Matriz ``(frames, features)`` ya normalizada, sin ``NaN``.
        valid: Máscara ``(frames,)``: ``False`` en los frames donde se perdió el
            torso y no se pudo reconstruir.
    """

    features: np.ndarray
    valid: np.ndarray

    @property
    def frames(self) -> int:
        """Cantidad de frames de la secuencia."""
        return int(self.features.shape[0])

    @property
    def valid_ratio(self) -> float:
        """Proporción de frames utilizables (0.0-1.0)."""
        if self.valid.size == 0:
            return 0.0
        return float(np.count_nonzero(self.valid) / self.valid.size)


def normalized_size(layout: FeatureLayout) -> int:
    """Largo del vector normalizado: el original más un indicador por parte."""
    return layout.size + len(layout.blocks)


class LandmarkNormalizer:
    """Aplica la normalización a secuencias de landmarks crudos."""

    def __init__(
        self,
        layout: FeatureLayout,
        *,
        aspect_ratio: float,
        max_gap_frames: int,
    ) -> None:
        """Args:
        layout: Layout del vector de features crudo.
        aspect_ratio: ``ancho / alto`` del frame de origen.
        max_gap_frames: Huecos de detección de hasta esta cantidad de frames se
            interpolan; los más largos se marcan como ausentes.
        """
        self._layout = layout
        self._aspect = float(aspect_ratio)
        self._max_gap = int(max_gap_frames)
        if not layout.has(BLOCK_POSE):
            raise ValueError(
                "La normalización necesita el bloque de pose: los hombros son el "
                "punto de referencia y la escala."
            )

    @classmethod
    def from_config(
        cls, config: Config, layout: FeatureLayout, *, aspect_ratio: float | None = None
    ) -> LandmarkNormalizer:
        """Construye el normalizador con la sección ``normalizacion`` del YAML.

        Args:
            aspect_ratio: Relación de aspecto real de la grabación. Si es
                ``None``, se usa la de la cámara configurada.
        """
        if aspect_ratio is None:
            aspect_ratio = float(config.require("camara.ancho")) / float(
                config.require("camara.alto")
            )
        return cls(
            layout,
            aspect_ratio=aspect_ratio,
            max_gap_frames=int(config.require("normalizacion.interpolar_huecos_max_frames")),
        )

    @property
    def output_size(self) -> int:
        """Largo del vector que produce este normalizador."""
        return normalized_size(self._layout)

    def normalize(self, landmarks: np.ndarray) -> NormalizedSequence:
        """Normaliza una secuencia de landmarks crudos.

        Args:
            landmarks: Matriz ``(frames, layout.size)`` tal como se guardó en el
                ``.npz``, con ``NaN`` en las partes no detectadas.

        Returns:
            La secuencia normalizada y su máscara de frames válidos.
        """
        if landmarks.ndim != 2 or landmarks.shape[1] != self._layout.size:
            raise ValueError(
                f"Se esperaba una matriz (frames, {self._layout.size}) y llegó "
                f"{landmarks.shape}."
            )

        datos = landmarks.astype(np.float32, copy=True)
        frames = datos.shape[0]
        if frames == 0:
            return NormalizedSequence(
                features=np.zeros((0, self.output_size), dtype=np.float32),
                valid=np.zeros(0, dtype=bool),
            )

        presencia = np.zeros((frames, len(self._layout.blocks)), dtype=np.float32)
        for indice, bloque in enumerate(self._layout.blocks):
            tramo = datos[:, bloque.slice]
            ausente = np.all(np.isnan(tramo), axis=1)
            _interpolate_gaps(tramo, ausente, self._max_gap)
            # Tras interpolar, un frame sigue ausente solo si el hueco era largo.
            presencia[:, indice] = (~np.all(np.isnan(tramo), axis=1)).astype(np.float32)
            datos[:, bloque.slice] = tramo

        centro, escala, valid = self._reference_frame(datos)
        self._apply_transform(datos, centro, escala)

        np.nan_to_num(datos, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return NormalizedSequence(
            features=np.concatenate([datos, presencia], axis=1, dtype=np.float32),
            valid=valid,
        )

    def _reference_frame(
        self, datos: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calcula centro y escala por frame a partir de los hombros."""
        bloque = self._layout.block(BLOCK_POSE)
        pose = datos[:, bloque.slice].reshape(-1, bloque.points, bloque.coords)

        izquierdo = pose[:, POSE_LEFT_SHOULDER, :3].copy()
        derecho = pose[:, POSE_RIGHT_SHOULDER, :3].copy()
        izquierdo[:, 0] *= self._aspect
        derecho[:, 0] *= self._aspect

        centro = (izquierdo + derecho) / 2.0
        distancia = np.linalg.norm(izquierdo[:, :2] - derecho[:, :2], axis=1)

        valid = np.isfinite(distancia) & (distancia > MIN_SHOULDER_DISTANCE)
        # Los frames sin referencia usable se dejan pasar sin escalar en vez de
        # dividir por cero; quedan marcados como inválidos.
        escala = np.where(valid, distancia, 1.0).astype(np.float32)
        centro = np.nan_to_num(centro, nan=0.0)
        return centro, escala, valid

    def _apply_transform(
        self, datos: np.ndarray, centro: np.ndarray, escala: np.ndarray
    ) -> None:
        """Aplica isotropía, centrado y escalado a todos los bloques, en sitio."""
        for bloque in self._layout.blocks:
            tramo = datos[:, bloque.slice].reshape(-1, bloque.points, bloque.coords)
            tramo[:, :, 0] *= self._aspect
            tramo[:, :, :3] -= centro[:, None, :]
            tramo[:, :, :3] /= escala[:, None, None]
            # La columna de visibilidad de la pose (índice 3) no se transforma:
            # es una probabilidad, no una coordenada.
            datos[:, bloque.slice] = tramo.reshape(datos.shape[0], -1)


def _interpolate_gaps(tramo: np.ndarray, ausente: np.ndarray, max_gap: int) -> None:
    """Rellena por interpolación lineal los huecos cortos de un bloque, en sitio.

    Args:
        tramo: Vista ``(frames, valores)`` del bloque.
        ausente: Máscara de frames sin detección para ese bloque.
        max_gap: Largo máximo de hueco que se interpola.
    """
    if max_gap <= 0 or not ausente.any() or ausente.all():
        return

    presentes = np.flatnonzero(~ausente)
    for inicio, fin in _gap_runs(ausente):
        largo = fin - inicio
        if largo > max_gap:
            continue
        antes = presentes[presentes < inicio]
        despues = presentes[presentes >= fin]
        if antes.size == 0 or despues.size == 0:
            # Hueco al principio o al final: no hay dos extremos que unir.
            continue
        anterior, siguiente = int(antes[-1]), int(despues[0])
        pesos = np.linspace(0.0, 1.0, siguiente - anterior + 1, dtype=np.float32)[1:-1]
        tramo[inicio:fin] = (
            tramo[anterior][None, :] * (1.0 - pesos)[:, None]
            + tramo[siguiente][None, :] * pesos[:, None]
        )


def _gap_runs(ausente: np.ndarray) -> list[tuple[int, int]]:
    """Devuelve los tramos ``[inicio, fin)`` de frames ausentes consecutivos."""
    tramos: list[tuple[int, int]] = []
    inicio: int | None = None
    for indice, falta in enumerate(ausente):
        if falta and inicio is None:
            inicio = indice
        elif not falta and inicio is not None:
            tramos.append((inicio, indice))
            inicio = None
    if inicio is not None:
        tramos.append((inicio, len(ausente)))
    return tramos
