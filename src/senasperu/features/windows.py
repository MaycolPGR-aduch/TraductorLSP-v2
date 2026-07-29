"""Extracción de ventanas de entrenamiento a partir de una repetición grabada.

Una repetición dura 3,5 s pero el modelo consume ventanas de 2 s, así que de cada
grabación salen varias muestras. Cuáles son válidas depende del tipo de seña:

- **Estática** (y la clase de reposo): sirve cualquier ventana; toda la toma es
  la seña.
- **Dinámica**: solo las ventanas que contienen el trazo casi entero. Una ventana
  que solo capta la mano subiendo, etiquetada como la seña, es exactamente lo que
  produce predicciones espurias en la app final.

El trazo se localiza por la velocidad de las muñecas. Como los landmarks ya están
escalados por la distancia entre hombros, la velocidad queda en *anchos de hombro
por segundo*, comparable entre personas y distancias de cámara.

Las ventanas se remuestrean siempre a la misma cantidad de frames, de modo que
una cámara a 25 FPS y otra a 30 FPS produzcan entradas idénticas para el modelo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from senasperu.config import Config
from senasperu.features.landmarks import POSE_LEFT_WRIST, POSE_RIGHT_WRIST
from senasperu.features.normalize import NormalizedSequence
from senasperu.features.vector import BLOCK_POSE, FeatureLayout

# Percentil que se usa como referencia de "velocidad alta" de una grabación.
# El máximo sería demasiado sensible a los saltos de landmark.
PEAK_PERCENTILE: float = 90.0


@dataclass(frozen=True, slots=True)
class Window:
    """Una ventana lista para alimentar al modelo.

    Attributes:
        features: Matriz ``(frames_por_ventana, features)``.
        start_frame: Frame de la grabación donde empieza la ventana.
        valid_ratio: Proporción de frames con torso detectado.
    """

    features: np.ndarray
    start_frame: int
    valid_ratio: float


class WindowExtractor:
    """Convierte una repetición normalizada en ventanas de entrenamiento."""

    def __init__(
        self,
        layout: FeatureLayout,
        *,
        window_seconds: float,
        frames_per_window: int,
        stride_frames: int,
        min_stroke_coverage: float,
        motion_threshold: float,
        smoothing_frames: int,
        min_valid_ratio: float,
        peak_fraction: float = 0.35,
        min_stroke_pct: float = 0.15,
        max_stroke_pct: float = 0.8,
    ) -> None:
        """Args:
        layout: Layout del vector crudo (para ubicar el bloque de pose).
        window_seconds: Duración real que abarca cada ventana.
        frames_per_window: Frames a los que se remuestrea cada ventana.
        stride_frames: Separación entre ventanas consecutivas.
        min_stroke_coverage: Fracción del trazo que debe caer dentro de la
            ventana para aceptarla (solo señas dinámicas).
        motion_threshold: Velocidad de muñeca que marca el trazo.
        smoothing_frames: Ventana del suavizado de la velocidad.
        min_valid_ratio: Mínimo de frames válidos para aceptar una ventana.
        peak_fraction: Fracción del pico de velocidad de la propia grabación
            que también hay que superar para considerar que hay trazo.
        """
        self._layout = layout
        self._window_seconds = float(window_seconds)
        self._frames_per_window = int(frames_per_window)
        self._stride = max(1, int(stride_frames))
        self._min_coverage = float(min_stroke_coverage)
        self._motion_threshold = float(motion_threshold)
        self._smoothing = max(1, int(smoothing_frames))
        self._min_valid_ratio = float(min_valid_ratio)
        self._peak_fraction = float(peak_fraction)
        self._min_stroke_pct = float(min_stroke_pct)
        self._max_stroke_pct = float(max_stroke_pct)

    @classmethod
    def from_config(cls, config: Config, layout: FeatureLayout) -> WindowExtractor:
        """Construye el extractor con la sección ``ventana`` del YAML."""
        return cls(
            layout,
            window_seconds=float(config.require("ventana.duracion_segundos")),
            frames_per_window=int(config.require("ventana.frames_por_ventana")),
            stride_frames=int(config.require("ventana.paso_frames")),
            min_stroke_coverage=float(config.require("ventana.cobertura_minima_trazo")),
            motion_threshold=float(config.require("ventana.umbral_movimiento")),
            smoothing_frames=int(config.require("ventana.suavizado_movimiento_frames")),
            min_valid_ratio=float(config.require("ventana.frames_validos_minimos")),
            peak_fraction=float(config.get("ventana.fraccion_pico_movimiento", 0.35)),
            min_stroke_pct=float(config.get("ventana.largo_trazo_min_pct", 0.15)),
            max_stroke_pct=float(config.get("ventana.largo_trazo_max_pct", 0.8)),
        )

    @property
    def frames_per_window(self) -> int:
        """Frames que entrega cada ventana."""
        return self._frames_per_window

    def extract(
        self, sequence: NormalizedSequence, fps: float, *, dynamic: bool
    ) -> list[Window]:
        """Extrae las ventanas válidas de una repetición.

        Args:
            sequence: Repetición ya normalizada.
            fps: FPS reales de la grabación.
            dynamic: ``True`` si la seña es dinámica (se exige contener el trazo).

        Returns:
            Las ventanas aceptadas, posiblemente vacía si la toma no sirve.
        """
        total = sequence.frames
        if total == 0 or fps <= 0:
            return []

        largo_bruto = max(2, int(round(self._window_seconds * fps)))
        if total < largo_bruto:
            # Grabación más corta que la ventana: se usa entera y se estira.
            largo_bruto = total

        # Si no hay un trazo plausible, la toma se trata como estática: todas
        # sus ventanas valen. Eso ocurre en dos casos, y en ambos es lo correcto:
        # la seña se sostiene (no hay movimiento) o el movimiento es continuo
        # durante toda la toma, como en un saludo, y entonces cualquier ventana
        # contiene la seña.
        trazo = self.active_segment(sequence, fps) if dynamic else None

        candidatos = list(range(0, max(1, total - largo_bruto + 1), self._stride))
        inicios = candidatos
        if trazo is not None:
            inicios = [
                inicio
                for inicio in candidatos
                if _coverage((inicio, inicio + largo_bruto), trazo) >= self._min_coverage
            ]
            if not inicios:
                # Ninguna ventana cubre el trazo lo suficiente. Descartar la
                # grabación entera sería perder datos que costó grabar, así que
                # se conserva la ventana que mejor lo contiene.
                mejor = max(
                    candidatos,
                    key=lambda inicio: _coverage((inicio, inicio + largo_bruto), trazo),
                )
                inicios = [mejor]
        return self._build(inicios, sequence, largo_bruto)

    def active_segment(
        self, sequence: NormalizedSequence, fps: float
    ) -> tuple[int, int] | None:
        """Localiza el tramo con movimiento de manos.

        Se toma el tramo **contiguo más largo** por encima del umbral, no del
        primer al último frame activo: los landmarks tiemblan, y un pico de
        ruido cerca de cada extremo bastaría para que "el trazo" fuese la
        grabación entera y ninguna ventana pudiera contenerlo.

        El umbral es el mayor entre el absoluto configurado y una fracción del
        pico de la propia grabación, para adaptarse a señas amplias y discretas.

        Returns:
            ``(inicio, fin)`` en índices de frame, o ``None`` si no hay
            movimiento por encima del umbral.
        """
        velocidad = self.wrist_speed(sequence, fps)
        if velocidad.size == 0:
            return None

        # Se usa el percentil 90 y no el máximo: cuando una muñeca entra al
        # encuadre, su landmark salta desde la posición extrapolada y produce un
        # pico aislado varias veces mayor que el movimiento real de la seña.
        # Con el máximo como referencia, ese artefacto secuestraba el umbral.
        pico = float(np.percentile(velocidad, PEAK_PERCENTILE))
        umbral = max(self._motion_threshold, self._peak_fraction * pico)
        activo = velocidad > umbral
        if not activo.any():
            return None

        tramo = _longest_run(activo)
        if tramo is None:
            return None
        inicio, fin = tramo
        # +1 en el fin porque la velocidad del índice i describe el paso i -> i+1.
        inicio, fin = int(inicio), int(fin) + 1

        # Solo se acepta el tramo si es plausible como trazo. Medido con
        # grabaciones reales: cuando la seña es oscilante (un saludo, por
        # ejemplo) o el encuadre deja el brazo fuera, lo que se detecta es
        # ruido, y filtrar por él descarta ventanas buenas. Ante la duda,
        # se prefiere no filtrar: no perder datos pesa más que afinar.
        largo = fin - inicio
        total = sequence.frames
        if largo < self._min_stroke_pct * total or largo > self._max_stroke_pct * total:
            return None
        return inicio, fin

    def wrist_speed(self, sequence: NormalizedSequence, fps: float) -> np.ndarray:
        """Velocidad suavizada de la muñeca más rápida, en anchos de hombro/s.

        Se suaviza la **posición** antes de derivar. Derivar una señal ruidosa
        amplifica el ruido: sin este paso, el temblor normal de los landmarks
        produce velocidades del orden del movimiento real.
        """
        bloque = self._layout.block(BLOCK_POSE)
        if sequence.frames < 2:
            return np.zeros(0, dtype=np.float32)

        pose = sequence.features[:, bloque.slice].reshape(-1, bloque.points, bloque.coords)
        munecas = pose[:, (POSE_LEFT_WRIST, POSE_RIGHT_WRIST), :2].astype(np.float32)
        suavizadas = np.stack(
            [
                _moving_average(munecas[:, mano, eje], self._smoothing)
                for mano in range(munecas.shape[1])
                for eje in range(2)
            ],
            axis=1,
        ).reshape(-1, munecas.shape[1], 2)

        desplazamiento = np.linalg.norm(np.diff(suavizadas, axis=0), axis=2)
        velocidad = (desplazamiento.max(axis=1) * float(fps)).astype(np.float32)
        # Mediana móvil antes del promedio: elimina los picos aislados que deja
        # un landmark al saltar de golpe, cosa que la media no consigue.
        return _moving_average(_rolling_median(velocidad, self._smoothing), self._smoothing)

    def _build(
        self, inicios: list[int], sequence: NormalizedSequence, largo_bruto: int
    ) -> list[Window]:
        ventanas: list[Window] = []
        for inicio in inicios:
            fin = min(inicio + largo_bruto, sequence.frames)
            valido = sequence.valid[inicio:fin]
            proporcion = float(np.count_nonzero(valido) / max(1, valido.size))
            if proporcion < self._min_valid_ratio:
                continue
            ventanas.append(
                Window(
                    features=_resample(
                        sequence.features[inicio:fin], self._frames_per_window
                    ),
                    start_frame=inicio,
                    valid_ratio=proporcion,
                )
            )
        return ventanas


def _coverage(ventana: tuple[int, int], trazo: tuple[int, int]) -> float:
    """Fracción del trazo contenida dentro de la ventana."""
    largo_trazo = trazo[1] - trazo[0]
    if largo_trazo <= 0:
        return 0.0
    solape = min(ventana[1], trazo[1]) - max(ventana[0], trazo[0])
    return max(0.0, solape) / largo_trazo


def _rolling_median(valores: np.ndarray, ventana: int) -> np.ndarray:
    """Mediana móvil centrada, conservando el largo del arreglo."""
    if ventana <= 1 or valores.size < 3:
        return valores
    ventana = min(ventana if ventana % 2 else ventana + 1, valores.size)
    borde = ventana // 2
    extendido = np.pad(valores, borde, mode="edge")
    tramos = np.lib.stride_tricks.sliding_window_view(extendido, ventana)
    return np.median(tramos, axis=1).astype(np.float32)


def _longest_run(activo: np.ndarray) -> tuple[int, int] | None:
    """Devuelve el tramo contiguo ``[inicio, fin]`` más largo de valores ``True``."""
    mejor: tuple[int, int] | None = None
    mejor_largo = 0
    inicio: int | None = None
    for indice, valor in enumerate(activo):
        if valor and inicio is None:
            inicio = indice
        elif not valor and inicio is not None:
            if indice - inicio > mejor_largo:
                mejor_largo, mejor = indice - inicio, (inicio, indice - 1)
            inicio = None
    if inicio is not None and len(activo) - inicio > mejor_largo:
        mejor = (inicio, len(activo) - 1)
    return mejor


def _moving_average(valores: np.ndarray, ventana: int) -> np.ndarray:
    """Media móvil centrada, conservando el largo del arreglo."""
    if ventana <= 1 or valores.size == 0:
        return valores
    ventana = min(ventana, valores.size)
    nucleo = np.ones(ventana, dtype=np.float32) / ventana
    return np.convolve(valores, nucleo, mode="same").astype(np.float32)


def resample_sequence(sequence: np.ndarray, target_frames: int) -> np.ndarray:
    """Remuestrea una secuencia a una cantidad fija de frames.

    Pública porque la inferencia en tiempo real debe aplicar exactamente el
    mismo remuestreo que el entrenamiento.
    """
    return _resample(sequence, target_frames)


def _resample(bloque: np.ndarray, objetivo: int) -> np.ndarray:
    """Remuestrea una secuencia a una cantidad fija de frames por interpolación.

    Es lo que hace que una cámara a 25 FPS y otra a 30 FPS produzcan entradas
    idénticas para el modelo.
    """
    origen = bloque.shape[0]
    if origen == objetivo:
        return bloque.astype(np.float32, copy=True)
    if origen == 1:
        return np.repeat(bloque.astype(np.float32), objetivo, axis=0)

    posiciones = np.linspace(0.0, origen - 1, objetivo, dtype=np.float32)
    piso = np.floor(posiciones).astype(np.int32)
    techo = np.minimum(piso + 1, origen - 1)
    peso = (posiciones - piso)[:, None]
    return (bloque[piso] * (1.0 - peso) + bloque[techo] * peso).astype(np.float32)
