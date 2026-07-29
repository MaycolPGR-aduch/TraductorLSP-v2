"""Pruebas del aumento de datos sobre landmarks normalizados."""

from __future__ import annotations

import numpy as np
import pytest

from senasperu.config import load_config
from senasperu.data.augment import POSE_MIRROR_PAIRS, WindowAugmenter
from senasperu.features.vector import (
    BLOCK_LEFT_HAND,
    BLOCK_POSE,
    BLOCK_RIGHT_HAND,
    build_layout,
    layout_from_config,
)

LAYOUT = build_layout(face_points=0)
FEATURES = LAYOUT.size + len(LAYOUT.blocks)


def aumentador(**cambios) -> WindowAugmenter:
    opciones = dict(
        rotation_degrees=0.0,
        scale_range=(1.0, 1.0),
        noise_std=0.0,
        time_jitter_pct=0.0,
        mirror_enabled=True,
        probability=1.0,
        seed=7,
    )
    opciones.update(cambios)
    return WindowAugmenter(LAYOUT, **opciones)


def ventana(frames: int = 12) -> np.ndarray:
    rng = np.random.default_rng(3)
    datos = rng.normal(0.0, 1.0, size=(frames, FEATURES)).astype(np.float32)
    datos[:, LAYOUT.size:] = 1.0  # indicadores de presencia
    return datos


def test_los_pares_de_espejado_de_la_pose_son_una_involucion() -> None:
    orden = np.arange(33)
    for i, j in POSE_MIRROR_PAIRS:
        orden[i], orden[j] = j, i
    assert np.array_equal(orden[orden], np.arange(33))


def test_espejar_dos_veces_devuelve_el_original() -> None:
    datos = ventana()
    doble = aumentador().mirror(aumentador().mirror(datos))
    assert np.allclose(doble, datos, atol=1e-6)


def test_el_espejado_intercambia_las_manos() -> None:
    """Reflejar a quien señó con la derecha produce a alguien señando con la izquierda."""
    datos = ventana()
    izquierda = LAYOUT.block(BLOCK_LEFT_HAND)
    derecha = LAYOUT.block(BLOCK_RIGHT_HAND)

    reflejada = aumentador().mirror(datos)

    original_izquierda = datos[:, izquierda.slice].reshape(-1, izquierda.points, 3).copy()
    nueva_derecha = reflejada[:, derecha.slice].reshape(-1, derecha.points, 3)
    original_izquierda[:, :, 0] *= -1.0
    assert np.allclose(nueva_derecha, original_izquierda, atol=1e-6)


def test_el_espejado_intercambia_los_pares_de_la_pose() -> None:
    datos = ventana()
    bloque = LAYOUT.block(BLOCK_POSE)
    original = datos[:, bloque.slice].reshape(-1, bloque.points, bloque.coords)
    reflejada = aumentador().mirror(datos)[:, bloque.slice].reshape(
        -1, bloque.points, bloque.coords
    )
    # El hombro izquierdo (11) debe pasar a ocupar el lugar del derecho (12).
    assert np.allclose(reflejada[:, 12, 1], original[:, 11, 1], atol=1e-6)
    assert np.allclose(reflejada[:, 12, 0], -original[:, 11, 0], atol=1e-6)


def test_el_espejado_no_altera_la_visibilidad() -> None:
    datos = ventana()
    bloque = LAYOUT.block(BLOCK_POSE)
    reflejada = aumentador().mirror(datos)
    visibilidad_original = datos[:, bloque.slice].reshape(-1, bloque.points, 4)[:, 11, 3]
    visibilidad_nueva = reflejada[:, bloque.slice].reshape(-1, bloque.points, 4)[:, 12, 3]
    assert np.allclose(visibilidad_nueva, visibilidad_original, atol=1e-6)


def test_la_rotacion_conserva_las_distancias() -> None:
    datos = ventana()
    bloque = LAYOUT.block(BLOCK_POSE)
    rotada = aumentador().rotate(datos, 30.0)

    def radios(x: np.ndarray) -> np.ndarray:
        puntos = x[:, bloque.slice].reshape(-1, bloque.points, bloque.coords)
        return np.linalg.norm(puntos[:, :, :2], axis=2)

    assert np.allclose(radios(rotada), radios(datos), atol=1e-4)


def test_rotar_cero_grados_no_cambia_nada() -> None:
    datos = ventana()
    assert np.allclose(aumentador().rotate(datos, 0.0), datos, atol=1e-6)


def test_el_ruido_no_toca_los_indicadores_de_presencia() -> None:
    datos = ventana()
    resultado = aumentador(noise_std=0.05).augment(datos)
    assert np.allclose(resultado[:, LAYOUT.size:], datos[:, LAYOUT.size:])
    assert not np.allclose(resultado[:, : LAYOUT.size], datos[:, : LAYOUT.size])


def test_la_escala_no_toca_los_indicadores_ni_la_visibilidad() -> None:
    datos = ventana()
    resultado = aumentador(scale_range=(2.0, 2.0)).augment(datos)
    bloque = LAYOUT.block(BLOCK_POSE)
    antes = datos[:, bloque.slice].reshape(-1, bloque.points, 4)
    despues = resultado[:, bloque.slice].reshape(-1, bloque.points, 4)
    assert np.allclose(despues[:, :, :3], antes[:, :, :3] * 2.0, atol=1e-5)
    assert np.allclose(despues[:, :, 3], antes[:, :, 3], atol=1e-6)
    assert np.allclose(resultado[:, LAYOUT.size:], datos[:, LAYOUT.size:])


def test_el_jitter_temporal_conserva_la_cantidad_de_frames() -> None:
    datos = ventana(24)
    for factor in (0.8, 1.25):
        resultado = aumentador().time_jitter(datos, factor)
        assert resultado.shape == datos.shape


def test_con_probabilidad_cero_no_se_aumenta_nada() -> None:
    datos = ventana()
    resultado = aumentador(probability=0.0, noise_std=1.0).augment(datos, mirrorable=True)
    assert np.allclose(resultado, datos)


def test_una_sena_no_espejable_nunca_se_refleja() -> None:
    datos = ventana()
    aug = aumentador(noise_std=0.0)
    for _ in range(20):
        resultado = aug.augment(datos, mirrorable=False)
        assert np.allclose(resultado, datos, atol=1e-6)


def test_el_aumento_no_modifica_la_ventana_original() -> None:
    datos = ventana()
    copia = datos.copy()
    aumentador(noise_std=0.1, rotation_degrees=10.0).augment(datos, mirrorable=True)
    assert np.array_equal(datos, copia)


def test_la_permutacion_facial_del_proyecto_es_valida() -> None:
    """Si no lo fuera, el aumentador desactivaría el espejado por seguridad."""
    config = load_config()
    aug = WindowAugmenter.from_config(config, layout_from_config(config))
    assert aug.mirror_enabled


def test_una_permutacion_facial_invalida_desactiva_el_espejado() -> None:
    config = load_config()
    layout = layout_from_config(config)
    aug = WindowAugmenter(
        layout,
        rotation_degrees=0.0,
        scale_range=(1.0, 1.0),
        noise_std=0.0,
        time_jitter_pct=0.0,
        mirror_enabled=True,
        probability=1.0,
        face_mirror=[0, 1, 2],  # largo incorrecto
    )
    assert not aug.mirror_enabled


def test_el_espejado_facial_del_proyecto_es_involucion() -> None:
    config = load_config()
    layout = layout_from_config(config)
    aug = WindowAugmenter.from_config(config, layout)
    features = layout.size + len(layout.blocks)
    datos = np.random.default_rng(1).normal(size=(6, features)).astype(np.float32)
    assert np.allclose(aug.mirror(aug.mirror(datos)), datos, atol=1e-6)


@pytest.mark.parametrize("frames", [2, 5, 48])
def test_el_jitter_funciona_con_ventanas_de_cualquier_largo(frames: int) -> None:
    datos = ventana(frames)
    assert aumentador().time_jitter(datos, 1.2).shape == datos.shape
