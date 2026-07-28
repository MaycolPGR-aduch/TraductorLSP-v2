"""Pruebas del vector de features que se guarda en el dataset."""

from __future__ import annotations

import numpy as np
import pytest

from senasperu.config import load_config
from senasperu.features.landmarks import HolisticResult
from senasperu.features.vector import (
    BLOCK_FACE,
    BLOCK_LEFT_HAND,
    BLOCK_POSE,
    BLOCK_RIGHT_HAND,
    build_layout,
    layout_from_config,
    to_feature_vector,
)


def test_tamano_del_layout_completo() -> None:
    layout = build_layout(face_points=26)
    # pose 33x4 + dos manos 21x3 + rostro 26x3
    assert layout.size == 132 + 63 + 63 + 78


def test_los_bloques_son_contiguos_y_sin_solapes() -> None:
    layout = build_layout(face_points=10)
    siguiente = 0
    for bloque in layout.blocks:
        assert bloque.start == siguiente
        siguiente += bloque.size
    assert siguiente == layout.size


def test_sin_rostro_no_hay_bloque_facial() -> None:
    layout = build_layout(face_points=0)
    assert not layout.has(BLOCK_FACE)
    assert layout.names == (BLOCK_POSE, BLOCK_LEFT_HAND, BLOCK_RIGHT_HAND)
    with pytest.raises(KeyError):
        layout.block(BLOCK_FACE)


def test_layout_desde_la_configuracion_del_proyecto() -> None:
    config = load_config()
    layout = layout_from_config(config)
    esperado = len(config.get("mediapipe.indices_rostro"))
    assert layout.block(BLOCK_FACE).points == esperado


def test_las_partes_ausentes_quedan_en_nan() -> None:
    """Un cero es una coordenada válida: lo ausente debe distinguirse de verdad."""
    layout = build_layout(face_points=0)
    vector = to_feature_vector(HolisticResult(), layout)
    assert vector.shape == (layout.size,)
    assert np.isnan(vector).all()


def test_cada_parte_va_a_su_tramo() -> None:
    layout = build_layout(face_points=0)
    pose = np.full((33, 4), 0.1, dtype=np.float32)
    izquierda = np.full((21, 3), 0.2, dtype=np.float32)

    vector = to_feature_vector(HolisticResult(pose=pose, left_hand=izquierda), layout)

    assert np.allclose(vector[layout.block(BLOCK_POSE).slice], 0.1)
    assert np.allclose(vector[layout.block(BLOCK_LEFT_HAND).slice], 0.2)
    assert np.isnan(vector[layout.block(BLOCK_RIGHT_HAND).slice]).all()


def test_una_parte_con_tamano_inesperado_se_marca_ausente() -> None:
    """Antes dejar el bloque vacío que escribir datos desalineados en el dataset."""
    layout = build_layout(face_points=0)
    mano_rara = np.zeros((5, 3), dtype=np.float32)

    vector = to_feature_vector(HolisticResult(right_hand=mano_rara), layout)

    assert np.isnan(vector[layout.block(BLOCK_RIGHT_HAND).slice]).all()


def test_el_vector_es_float32() -> None:
    layout = build_layout(face_points=4)
    vector = to_feature_vector(HolisticResult(face=np.zeros((4, 3), dtype=np.float32)), layout)
    assert vector.dtype == np.float32
