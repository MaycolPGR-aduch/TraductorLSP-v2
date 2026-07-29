"""Pruebas del buffer de ventana deslizante de la inferencia en vivo."""

from __future__ import annotations

import numpy as np

from senasperu.features.landmarks import (
    POSE_LEFT_SHOULDER,
    POSE_RIGHT_SHOULDER,
    HolisticResult,
)
from senasperu.features.normalize import LandmarkNormalizer, normalized_size
from senasperu.features.stream_buffer import StreamWindowBuffer
from senasperu.features.vector import build_layout

LAYOUT = build_layout(face_points=0)
FRAMES_POR_VENTANA = 48


def buffer(**cambios) -> StreamWindowBuffer:
    opciones = dict(
        window_seconds=2.0,
        frames_per_window=FRAMES_POR_VENTANA,
        stride_frames=8,
        fps=30.0,
    )
    opciones.update(cambios)
    normalizador = LandmarkNormalizer(LAYOUT, aspect_ratio=4 / 3, max_gap_frames=3)
    return StreamWindowBuffer(LAYOUT, normalizador, **opciones)


def resultado() -> HolisticResult:
    pose = np.zeros((33, 4), dtype=np.float32)
    pose[:, 3] = 1.0
    pose[POSE_LEFT_SHOULDER, :2] = (0.4, 0.5)
    pose[POSE_RIGHT_SHOULDER, :2] = (0.6, 0.5)
    return HolisticResult(pose=pose)


def test_no_entrega_ventanas_hasta_llenarse() -> None:
    buf = buffer()
    for _ in range(buf.capacity - 1):
        assert buf.push(resultado()) is None


def test_la_primera_ventana_sale_al_llenarse() -> None:
    buf = buffer()
    ventana = None
    for _ in range(buf.capacity):
        ventana = buf.push(resultado())
    assert ventana is not None
    assert ventana.shape == (FRAMES_POR_VENTANA, normalized_size(LAYOUT))


def test_despues_entrega_una_ventana_cada_paso() -> None:
    buf = buffer(stride_frames=8)
    for _ in range(buf.capacity):
        buf.push(resultado())

    salidas = [buf.push(resultado()) is not None for _ in range(24)]
    assert sum(salidas) == 3, "con paso 8 deben salir 3 ventanas en 24 frames"


def test_el_progreso_de_llenado_es_informativo() -> None:
    buf = buffer()
    assert buf.ready_ratio == 0.0
    for _ in range(buf.capacity // 2):
        buf.push(resultado())
    assert 0.4 < buf.ready_ratio < 0.6
    for _ in range(buf.capacity):
        buf.push(resultado())
    assert buf.ready_ratio == 1.0


def test_clear_vacia_el_buffer() -> None:
    buf = buffer()
    for _ in range(buf.capacity):
        buf.push(resultado())
    buf.clear()
    assert buf.filled == 0
    assert buf.push(resultado()) is None


def test_la_ventana_no_tiene_nan() -> None:
    """Aunque falten manos y rostro: el modelo no puede recibir NaN."""
    buf = buffer()
    ventana = None
    for _ in range(buf.capacity):
        ventana = buf.push(resultado())
    assert ventana is not None
    assert np.isfinite(ventana).all()
