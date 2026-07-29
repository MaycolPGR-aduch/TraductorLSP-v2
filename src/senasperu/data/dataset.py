"""Carga del dataset y construcción de los conjuntos de entrenamiento.

**Los splits son por sesión completa, nunca por repetición.** Dos repeticiones de
la misma sesión comparten iluminación, ropa, encuadre y el estado del señante ese
día: repartirlas entre train y test hace que el modelo reconozca la sesión, no la
seña, y da una precisión inflada que se derrumba con usuarios reales.

Sin PyTorch: aquí solo hay NumPy. El entrenamiento envuelve estos arreglos.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from senasperu.config import Config
from senasperu.data.dataset_writer import FILE_PATTERN
from senasperu.features.normalize import LandmarkNormalizer
from senasperu.features.vector import FeatureLayout, layout_from_config
from senasperu.features.windows import WindowExtractor
from senasperu.vocabulary import Sign, load_vocabulary

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Recording:
    """Una repetición en disco, sin cargar todavía."""

    path: Path
    label: str
    person: str
    session: int
    repetition: int

    @property
    def session_key(self) -> tuple[str, int]:
        """Clave de agrupación para los splits: persona + sesión."""
        return (self.person, self.session)


@dataclass(frozen=True, slots=True)
class WindowSet:
    """Ventanas listas para el modelo.

    Attributes:
        features: ``(n, frames, features)``.
        labels: ``(n,)`` con el índice de clase de cada ventana.
        mirrorable: ``(n,)`` indicando si la seña admite espejado.
        session_keys: Sesión de origen de cada ventana (para diagnóstico).
    """

    features: np.ndarray
    labels: np.ndarray
    mirrorable: np.ndarray
    session_keys: tuple[tuple[str, int], ...]

    def __len__(self) -> int:
        return int(self.features.shape[0])

    @property
    def counts_by_label(self) -> dict[int, int]:
        """Cantidad de ventanas por clase."""
        valores, cuentas = np.unique(self.labels, return_counts=True)
        return {int(v): int(c) for v, c in zip(valores, cuentas)}


def scan_recordings(root: Path) -> list[Recording]:
    """Recorre ``dataset/raw`` y lista las repeticiones encontradas."""
    grabaciones: list[Recording] = []
    if not root.is_dir():
        return grabaciones
    for archivo in sorted(root.glob("*/*.npz")):
        datos = FILE_PATTERN.match(archivo.stem)
        if datos is None:
            logger.warning("Nombre de archivo inesperado, se ignora: %s", archivo)
            continue
        grabaciones.append(
            Recording(
                path=archivo,
                label=archivo.parent.name,
                person=datos["person"],
                session=int(datos["session"]),
                repetition=int(datos["repetition"]),
            )
        )
    return grabaciones


def split_by_session(
    recordings: list[Recording], *, test_ratio: float, seed: int
) -> tuple[list[Recording], list[Recording]]:
    """Reparte las grabaciones en train y test **por sesión completa**.

    Args:
        recordings: Todas las grabaciones disponibles.
        test_ratio: Proporción objetivo de grabaciones para test.
        seed: Semilla, para que el split sea reproducible.

    Returns:
        Tupla ``(train, test)``.

    Raises:
        ValueError: Si hay una sola sesión, porque entonces no existe ningún
            split honesto posible.
    """
    por_sesion: dict[tuple[str, int], list[Recording]] = defaultdict(list)
    for grabacion in recordings:
        por_sesion[grabacion.session_key].append(grabacion)

    sesiones = sorted(por_sesion)
    if len(sesiones) < 2:
        raise ValueError(
            "Hay una sola sesión en el dataset: no se puede evaluar honestamente. "
            "Graba al menos una sesión más, otro día y con otras condiciones."
        )

    rng = np.random.default_rng(seed)
    orden = list(rng.permutation(len(sesiones)))
    objetivo = test_ratio * len(recordings)

    test: list[Recording] = []
    sesiones_test: set[tuple[str, int]] = set()
    for indice in orden:
        if len(test) >= objetivo:
            break
        clave = sesiones[indice]
        # No dejar ninguna clase fuera de train por culpa del split.
        candidato = test + por_sesion[clave]
        if _labels_left_out(recordings, candidato):
            continue
        test = candidato
        sesiones_test.add(clave)

    train = [g for g in recordings if g.session_key not in sesiones_test]
    if not test:
        raise ValueError(
            "No se pudo apartar ninguna sesión para test sin dejar clases sin "
            "entrenar. Graba más sesiones antes de evaluar."
        )
    logger.info(
        "Split por sesión: %s grabaciones de entrenamiento (%s sesiones), "
        "%s de prueba (%s sesiones: %s)",
        len(train),
        len(sesiones) - len(sesiones_test),
        len(test),
        len(sesiones_test),
        sorted(sesiones_test),
    )
    return train, test


def _labels_left_out(todas: list[Recording], test: list[Recording]) -> bool:
    """``True`` si apartar ``test`` dejaría alguna clase sin ejemplos de train."""
    en_test = {g.path for g in test}
    etiquetas_train = {g.label for g in todas if g.path not in en_test}
    return etiquetas_train != {g.label for g in todas}


def build_window_set(
    recordings: list[Recording],
    config: Config,
    *,
    vocabulary: tuple[Sign, ...] | None = None,
    layout: FeatureLayout | None = None,
) -> WindowSet:
    """Carga las grabaciones y las convierte en ventanas de entrenamiento.

    Args:
        recordings: Grabaciones a cargar.
        config: Configuración del proyecto.
        vocabulary: Vocabulario; si es ``None``, se lee de la configuración.
        layout: Layout del vector; si es ``None``, se deriva de la configuración.

    Returns:
        El conjunto de ventanas resultante.
    """
    vocabulary = vocabulary or load_vocabulary(config)
    layout = layout or layout_from_config(config)
    extractor = WindowExtractor.from_config(config, layout)
    por_id = {sign.id: sign for sign in vocabulary}

    ventanas: list[np.ndarray] = []
    etiquetas: list[int] = []
    espejables: list[bool] = []
    sesiones: list[tuple[str, int]] = []
    descartadas = 0

    for grabacion in recordings:
        sign = por_id.get(grabacion.label)
        if sign is None:
            logger.warning(
                "La grabación %s tiene una seña que no está en el vocabulario; se ignora.",
                grabacion.path.name,
            )
            continue

        with np.load(grabacion.path, allow_pickle=False) as datos:
            landmarks = datos["landmarks"]
            fps = float(datos["fps"])
            ancho = int(datos["frame_width"]) if "frame_width" in datos else 0
            alto = int(datos["frame_height"]) if "frame_height" in datos else 0

        aspecto = (ancho / alto) if ancho > 0 and alto > 0 else None
        normalizador = LandmarkNormalizer.from_config(config, layout, aspect_ratio=aspecto)
        secuencia = normalizador.normalize(landmarks)

        extraidas = extractor.extract(secuencia, fps, dynamic=not sign.is_static)
        if not extraidas:
            descartadas += 1
            continue
        for ventana in extraidas:
            ventanas.append(ventana.features)
            etiquetas.append(sign.index)
            espejables.append(sign.mirrorable)
            sesiones.append(grabacion.session_key)

    if descartadas:
        logger.warning(
            "%s grabaciones no produjeron ninguna ventana utilizable "
            "(torso perdido o trazo no detectado).",
            descartadas,
        )
    if not ventanas:
        return WindowSet(
            features=np.zeros((0, extractor.frames_per_window, 0), dtype=np.float32),
            labels=np.zeros(0, dtype=np.int64),
            mirrorable=np.zeros(0, dtype=bool),
            session_keys=(),
        )

    return WindowSet(
        features=np.stack(ventanas).astype(np.float32),
        labels=np.asarray(etiquetas, dtype=np.int64),
        mirrorable=np.asarray(espejables, dtype=bool),
        session_keys=tuple(sesiones),
    )
