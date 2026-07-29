"""Pruebas de la normalización de landmarks."""

from __future__ import annotations

import numpy as np
import pytest

from senasperu.features.landmarks import POSE_LEFT_SHOULDER, POSE_RIGHT_SHOULDER
from senasperu.features.normalize import LandmarkNormalizer, normalized_size
from senasperu.features.vector import (
    BLOCK_LEFT_HAND,
    BLOCK_POSE,
    build_layout,
)

LAYOUT = build_layout(face_points=0)


# Desplazamientos fijos de cada landmark respecto al cuerpo, en unidades de
# ancho de hombros. Así la secuencia sintética es un cuerpo rígido: al moverlo o
# escalarlo, la geometría relativa (que es lo que el modelo debe ver) no cambia.
_OFFSETS_POSE = np.linspace(-1.0, 1.0, 33 * 2, dtype=np.float32).reshape(33, 2)
_OFFSETS_MANO = np.linspace(-0.4, 0.4, 21 * 3, dtype=np.float32).reshape(21, 3)


def secuencia(frames: int = 10, *, centro=(0.5, 0.5), ancho_hombros=0.2) -> np.ndarray:
    """Secuencia sintética: el mismo cuerpo rígido, movido y escalado."""
    datos = np.full((frames, LAYOUT.size), np.nan, dtype=np.float32)

    pose = np.zeros((frames, 33, 4), dtype=np.float32)
    pose[:, :, 3] = 1.0
    pose[:, :, :2] = np.asarray(centro, dtype=np.float32) + _OFFSETS_POSE * ancho_hombros
    pose[:, POSE_LEFT_SHOULDER, :2] = (centro[0] - ancho_hombros / 2, centro[1])
    pose[:, POSE_RIGHT_SHOULDER, :2] = (centro[0] + ancho_hombros / 2, centro[1])
    datos[:, LAYOUT.block(BLOCK_POSE).slice] = pose.reshape(frames, -1)

    mano = np.zeros((frames, 21, 3), dtype=np.float32)
    mano[:, :, :2] = np.asarray(centro, dtype=np.float32) + _OFFSETS_MANO[:, :2] * ancho_hombros
    mano[:, :, 2] = _OFFSETS_MANO[:, 2] * ancho_hombros
    datos[:, LAYOUT.block(BLOCK_LEFT_HAND).slice] = mano.reshape(frames, -1)
    return datos


def normalizador(aspecto: float = 1.0, max_gap: int = 3) -> LandmarkNormalizer:
    return LandmarkNormalizer(LAYOUT, aspect_ratio=aspecto, max_gap_frames=max_gap)


def test_tamano_de_salida_incluye_los_indicadores_de_presencia() -> None:
    resultado = normalizador().normalize(secuencia())
    assert resultado.features.shape[1] == normalized_size(LAYOUT)
    assert resultado.features.shape[1] == LAYOUT.size + len(LAYOUT.blocks)


def test_la_salida_no_tiene_nan() -> None:
    """El modelo no puede recibir NaN: los huecos largos se rellenan con ceros."""
    datos = secuencia()
    datos[:, LAYOUT.block(BLOCK_LEFT_HAND).slice] = np.nan
    resultado = normalizador().normalize(datos)
    assert np.isfinite(resultado.features).all()


def test_es_invariante_a_la_posicion_en_el_cuadro() -> None:
    """La misma seña en otra parte de la imagen debe dar el mismo vector."""
    izquierda = normalizador().normalize(secuencia(centro=(0.3, 0.4)))
    derecha = normalizador().normalize(secuencia(centro=(0.7, 0.6)))
    assert np.allclose(izquierda.features, derecha.features, atol=1e-5)


def test_es_invariante_a_la_distancia_de_la_camara() -> None:
    """Más cerca o más lejos cambia el tamaño, no la seña."""
    cerca = normalizador().normalize(secuencia(ancho_hombros=0.4))
    lejos = normalizador().normalize(secuencia(ancho_hombros=0.15))
    assert np.allclose(cerca.features, lejos.features, atol=1e-5)


def test_los_hombros_quedan_a_distancia_uno() -> None:
    resultado = normalizador().normalize(secuencia())
    bloque = LAYOUT.block(BLOCK_POSE)
    pose = resultado.features[:, bloque.slice].reshape(-1, bloque.points, bloque.coords)
    distancia = np.linalg.norm(
        pose[:, POSE_LEFT_SHOULDER, :2] - pose[:, POSE_RIGHT_SHOULDER, :2], axis=1
    )
    assert np.allclose(distancia, 1.0, atol=1e-5)


def test_corrige_la_relacion_de_aspecto() -> None:
    """En 4:3, un desplazamiento horizontal y otro vertical iguales en píxeles
    llegan con números distintos y hay que compensarlo."""
    datos = secuencia()
    sin_correccion = LandmarkNormalizer(LAYOUT, aspect_ratio=1.0, max_gap_frames=3)
    con_correccion = LandmarkNormalizer(LAYOUT, aspect_ratio=4 / 3, max_gap_frames=3)
    assert not np.allclose(
        sin_correccion.normalize(datos).features, con_correccion.normalize(datos).features
    )


def test_interpola_los_huecos_cortos() -> None:
    datos = secuencia(frames=12)
    bloque = LAYOUT.block(BLOCK_LEFT_HAND)
    datos[5:7, bloque.slice] = np.nan  # hueco de 2 frames

    resultado = normalizador(max_gap=3).normalize(datos)
    presencia = resultado.features[:, LAYOUT.size + list(LAYOUT.names).index(BLOCK_LEFT_HAND)]
    assert presencia[5] == 1.0 and presencia[6] == 1.0, "el hueco corto debe interpolarse"


def test_no_interpola_los_huecos_largos() -> None:
    datos = secuencia(frames=15)
    bloque = LAYOUT.block(BLOCK_LEFT_HAND)
    datos[4:12, bloque.slice] = np.nan  # hueco de 8 frames

    resultado = normalizador(max_gap=3).normalize(datos)
    presencia = resultado.features[:, LAYOUT.size + list(LAYOUT.names).index(BLOCK_LEFT_HAND)]
    assert presencia[7] == 0.0, "un hueco largo debe quedar marcado como ausente"
    assert presencia[0] == 1.0


def test_los_frames_sin_hombros_se_marcan_invalidos() -> None:
    datos = secuencia(frames=8)
    datos[3:5, LAYOUT.block(BLOCK_POSE).slice] = np.nan

    resultado = normalizador(max_gap=0).normalize(datos)
    assert not resultado.valid[3] and not resultado.valid[4]
    assert resultado.valid[0]
    assert resultado.valid_ratio == pytest.approx(6 / 8)


def test_una_secuencia_vacia_no_revienta() -> None:
    resultado = normalizador().normalize(np.zeros((0, LAYOUT.size), dtype=np.float32))
    assert resultado.frames == 0
    assert resultado.valid_ratio == 0.0


def test_forma_incorrecta_da_error_claro() -> None:
    with pytest.raises(ValueError, match="frames"):
        normalizador().normalize(np.zeros((5, 3), dtype=np.float32))


def test_sin_pose_no_se_puede_normalizar() -> None:
    from senasperu.features.vector import FeatureLayout

    with pytest.raises(ValueError, match="hombros"):
        LandmarkNormalizer(FeatureLayout(blocks=()), aspect_ratio=1.0, max_gap_frames=3)
