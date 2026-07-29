"""Aumento de datos sobre landmarks normalizados.

Trabajamos sobre coordenadas, no sobre imágenes: las transformaciones son
baratas y no introducen artefactos de compresión ni de interpolación de píxeles.

Todas se aplican sobre ventanas ya normalizadas, donde el origen es el punto
medio de los hombros y la unidad es la distancia entre hombros.

El espejado horizontal merece cuidado aparte: reflejar a una persona que señó
con la derecha produce a alguien señando con la izquierda, así que además de
negar ``x`` hay que **intercambiar** las partes izquierda y derecha (manos,
pares de la pose y puntos simétricos del rostro). Negar ``x`` sin intercambiar
generaría un cuerpo imposible. Por eso solo se aplica a las señas marcadas como
``espejable`` en el vocabulario.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from senasperu.config import Config
from senasperu.features.vector import (
    BLOCK_FACE,
    BLOCK_LEFT_HAND,
    BLOCK_POSE,
    BLOCK_RIGHT_HAND,
    FeatureLayout,
)

logger = logging.getLogger(__name__)

# Pares izquierda/derecha de la pose de MediaPipe. El resto de los puntos
# (la nariz) se mapea a sí mismo. Verificados con grabaciones reales.
POSE_MIRROR_PAIRS: tuple[tuple[int, int], ...] = (
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
    (17, 18), (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
)


class WindowAugmenter:
    """Aplica aumentos aleatorios a una ventana normalizada."""

    def __init__(
        self,
        layout: FeatureLayout,
        *,
        rotation_degrees: float,
        scale_range: tuple[float, float],
        noise_std: float,
        time_jitter_pct: float,
        mirror_enabled: bool,
        probability: float,
        face_mirror: Sequence[int] | None = None,
        seed: int = 0,
    ) -> None:
        """Args:
        layout: Layout del vector crudo (el normalizado añade las presencias).
        rotation_degrees: Rotación máxima, en grados, en el plano de la imagen.
        scale_range: Rango multiplicativo de escala.
        noise_std: Desviación del ruido gaussiano, en unidades de hombro.
        time_jitter_pct: Variación máxima de velocidad, en porcentaje.
        mirror_enabled: Si el espejado está habilitado globalmente.
        probability: Probabilidad de aplicar aumentos a cada muestra.
        face_mirror: Permutación de espejado del subconjunto facial.
        seed: Semilla del generador aleatorio.
        """
        self._layout = layout
        self._rotation = float(rotation_degrees)
        self._scale_range = (float(scale_range[0]), float(scale_range[1]))
        self._noise_std = float(noise_std)
        self._time_jitter = float(time_jitter_pct) / 100.0
        self._probability = float(probability)
        self._rng = np.random.default_rng(seed)
        self._face_mirror = self._validate_face_mirror(face_mirror)
        self._mirror_enabled = mirror_enabled and (
            not layout.has(BLOCK_FACE) or self._face_mirror is not None
        )
        if mirror_enabled and not self._mirror_enabled:
            logger.warning(
                "El espejado horizontal queda desactivado: la permutación facial "
                "'mediapipe.espejo_rostro' no es válida para los índices configurados."
            )

    @classmethod
    def from_config(cls, config: Config, layout: FeatureLayout) -> WindowAugmenter:
        """Construye el aumentador con la sección ``entrenamiento.augmentation``."""
        rango = config.get("entrenamiento.augmentation.escala_rango", [0.9, 1.1])
        return cls(
            layout,
            rotation_degrees=float(
                config.require("entrenamiento.augmentation.rotacion_grados_max")
            ),
            scale_range=(float(rango[0]), float(rango[1])),
            noise_std=float(config.require("entrenamiento.augmentation.ruido_gaussiano_std")),
            time_jitter_pct=float(
                config.require("entrenamiento.augmentation.jitter_temporal_pct")
            ),
            mirror_enabled=bool(
                config.get("entrenamiento.augmentation.espejado_horizontal", False)
            ),
            probability=float(
                config.require("entrenamiento.augmentation.probabilidad_augmentation")
            ),
            face_mirror=config.get("mediapipe.espejo_rostro"),
            seed=int(config.get("entrenamiento.semilla", 0)),
        )

    @property
    def mirror_enabled(self) -> bool:
        """``True`` si el espejado se puede aplicar."""
        return self._mirror_enabled

    def augment(self, window: np.ndarray, *, mirrorable: bool = False) -> np.ndarray:
        """Devuelve una versión aumentada de la ventana.

        Args:
            window: Ventana normalizada ``(frames, features)``.
            mirrorable: Si la seña admite espejado horizontal.

        Returns:
            Una ventana nueva; la original no se modifica.
        """
        if self._rng.random() > self._probability:
            return window.astype(np.float32, copy=True)

        salida = window.astype(np.float32, copy=True)
        if mirrorable and self._mirror_enabled and self._rng.random() < 0.5:
            salida = self.mirror(salida)
        if self._rotation > 0:
            angulo = float(self._rng.uniform(-self._rotation, self._rotation))
            salida = self.rotate(salida, angulo)
        # Se compara el factor con 1.0, no los extremos del rango entre sí: un
        # rango degenerado como [1.2, 1.2] es una escala constante legítima y
        # antes se ignoraba en silencio.
        factor = float(self._rng.uniform(*self._scale_range))
        if not np.isclose(factor, 1.0):
            self._scale_coords(salida, factor)
        if self._noise_std > 0:
            self._add_noise(salida)
        if self._time_jitter > 0:
            factor = float(self._rng.uniform(1.0 - self._time_jitter, 1.0 + self._time_jitter))
            salida = self.time_jitter(salida, factor)
        return salida

    # -- Transformaciones individuales -------------------------------------
    def mirror(self, window: np.ndarray) -> np.ndarray:
        """Refleja horizontalmente, intercambiando izquierda y derecha."""
        salida = window.copy()
        for bloque in self._layout.blocks:
            tramo = salida[:, bloque.slice].reshape(-1, bloque.points, bloque.coords)
            tramo[:, :, 0] *= -1.0
            if bloque.name == BLOCK_POSE:
                orden = np.arange(bloque.points)
                for i, j in POSE_MIRROR_PAIRS:
                    if i < bloque.points and j < bloque.points:
                        orden[i], orden[j] = j, i
                tramo = tramo[:, orden]
            elif bloque.name == BLOCK_FACE and self._face_mirror is not None:
                tramo = tramo[:, self._face_mirror]
            salida[:, bloque.slice] = tramo.reshape(salida.shape[0], -1)

        if self._layout.has(BLOCK_LEFT_HAND) and self._layout.has(BLOCK_RIGHT_HAND):
            izquierda = self._layout.block(BLOCK_LEFT_HAND)
            derecha = self._layout.block(BLOCK_RIGHT_HAND)
            copia_izquierda = salida[:, izquierda.slice].copy()
            salida[:, izquierda.slice] = salida[:, derecha.slice]
            salida[:, derecha.slice] = copia_izquierda
            self._swap_presence(salida, BLOCK_LEFT_HAND, BLOCK_RIGHT_HAND)
        return salida

    def rotate(self, window: np.ndarray, degrees: float) -> np.ndarray:
        """Rota el plano ``x, y`` alrededor del punto medio de los hombros."""
        salida = window.copy()
        angulo = np.deg2rad(degrees)
        coseno, seno = np.cos(angulo), np.sin(angulo)
        for bloque in self._layout.blocks:
            tramo = salida[:, bloque.slice].reshape(-1, bloque.points, bloque.coords)
            x = tramo[:, :, 0].copy()
            y = tramo[:, :, 1].copy()
            tramo[:, :, 0] = x * coseno - y * seno
            tramo[:, :, 1] = x * seno + y * coseno
            salida[:, bloque.slice] = tramo.reshape(salida.shape[0], -1)
        return salida

    def time_jitter(self, window: np.ndarray, factor: float) -> np.ndarray:
        """Cambia la velocidad de ejecución conservando la cantidad de frames.

        Un factor mayor que 1 simula una seña más lenta (se ve un tramo más
        corto del gesto); menor que 1, una más rápida.
        """
        frames = window.shape[0]
        if frames < 2 or factor <= 0:
            return window
        largo = max(2.0, (frames - 1) / factor)
        posiciones = np.clip(
            np.linspace(0.0, largo, frames, dtype=np.float32), 0.0, frames - 1
        )
        piso = np.floor(posiciones).astype(np.int32)
        techo = np.minimum(piso + 1, frames - 1)
        peso = (posiciones - piso)[:, None]
        return (window[piso] * (1.0 - peso) + window[techo] * peso).astype(np.float32)

    def _scale_coords(self, window: np.ndarray, factor: float) -> None:
        """Escala las coordenadas, dejando visibilidad y presencias intactas."""
        for bloque in self._layout.blocks:
            tramo = window[:, bloque.slice].reshape(-1, bloque.points, bloque.coords)
            tramo[:, :, :3] *= factor
            window[:, bloque.slice] = tramo.reshape(window.shape[0], -1)

    def _add_noise(self, window: np.ndarray) -> None:
        """Suma ruido gaussiano a las coordenadas, no a los indicadores."""
        for bloque in self._layout.blocks:
            tramo = window[:, bloque.slice].reshape(-1, bloque.points, bloque.coords)
            tramo[:, :, :3] += self._rng.normal(
                0.0, self._noise_std, size=tramo[:, :, :3].shape
            ).astype(np.float32)
            window[:, bloque.slice] = tramo.reshape(window.shape[0], -1)

    def _swap_presence(self, window: np.ndarray, primero: str, segundo: str) -> None:
        """Intercambia los indicadores de presencia de dos bloques."""
        nombres = list(self._layout.names)
        base = self._layout.size
        i, j = base + nombres.index(primero), base + nombres.index(segundo)
        if max(i, j) < window.shape[1]:
            window[:, [i, j]] = window[:, [j, i]]

    def _validate_face_mirror(self, face_mirror: Sequence[int] | None) -> np.ndarray | None:
        """Comprueba que la permutación facial sea usable; ``None`` si no lo es."""
        if face_mirror is None or not self._layout.has(BLOCK_FACE):
            return None
        permutacion = np.asarray(list(face_mirror), dtype=np.int32)
        puntos = self._layout.block(BLOCK_FACE).points
        if permutacion.size != puntos:
            return None
        if permutacion.min(initial=0) < 0 or permutacion.max(initial=0) >= puntos:
            return None
        # Debe ser una involución: el simétrico del simétrico es uno mismo.
        if not np.array_equal(permutacion[permutacion], np.arange(puntos)):
            return None
        return permutacion
