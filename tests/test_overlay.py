"""Pruebas del dibujo de landmarks (sin cámara: se usan imágenes en memoria)."""

from __future__ import annotations

import numpy as np

from senasperu.features.landmarks import POSE_UPPER_BODY, HolisticResult
from senasperu.features.overlay import LandmarkOverlay, _to_pixels


def imagen_negra(alto: int = 120, ancho: int = 160) -> np.ndarray:
    return np.zeros((alto, ancho, 3), dtype=np.uint8)


def test_to_pixels_convierte_y_recorta() -> None:
    puntos = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5], [-2.0, 3.0]], dtype=np.float32)
    pixeles = _to_pixels(puntos, ancho=100, alto=50)
    assert pixeles.tolist() == [[0, 0], [99, 49], [50, 25], [0, 49]]


def test_la_pose_ignora_caderas_y_piernas() -> None:
    """Los landmarks fuera del torso se extrapolan fuera del cuadro: no se dibujan."""
    pose = np.zeros((33, 4), dtype=np.float32)
    pose[:, 3] = 1.0
    for indice in range(23, 33):  # caderas y piernas, todas en el centro
        pose[indice, :2] = (0.5, 0.5)

    imagen = imagen_negra()
    LandmarkOverlay(draw_face=False).draw(imagen, HolisticResult(pose=pose))

    centro = imagen[55:65, 75:85]
    assert not centro.any(), "se dibujaron landmarks de piernas o caderas"


def test_la_pose_dibuja_el_torso() -> None:
    pose = np.zeros((33, 4), dtype=np.float32)
    pose[:, 3] = 1.0
    for indice in POSE_UPPER_BODY:
        pose[indice, :2] = (0.5, 0.5)

    imagen = imagen_negra()
    LandmarkOverlay(draw_face=False).draw(imagen, HolisticResult(pose=pose))

    assert imagen[55:65, 75:85].any(), "no se dibujó el torso"


def test_las_manos_se_dibujan_con_colores_distintos() -> None:
    mano = np.full((21, 3), 0.25, dtype=np.float32)
    otra = np.full((21, 3), 0.75, dtype=np.float32)

    imagen = imagen_negra()
    LandmarkOverlay(draw_face=False).draw(
        imagen, HolisticResult(left_hand=mano, right_hand=otra)
    )

    color_izquierda = imagen[30, 40]
    color_derecha = imagen[90, 120]
    assert color_izquierda.any() and color_derecha.any()
    assert not np.array_equal(color_izquierda, color_derecha), (
        "ambas manos se dibujan del mismo color: no se distinguen en pantalla"
    )


def test_sin_landmarks_la_imagen_no_cambia() -> None:
    imagen = imagen_negra()
    LandmarkOverlay().draw(imagen, HolisticResult())
    assert not imagen.any()


def test_el_rostro_se_puede_desactivar() -> None:
    rostro = np.full((26, 3), 0.5, dtype=np.float32)

    con_rostro = imagen_negra()
    LandmarkOverlay(draw_face=True).draw(con_rostro, HolisticResult(face=rostro))
    sin_rostro = imagen_negra()
    LandmarkOverlay(draw_face=False).draw(sin_rostro, HolisticResult(face=rostro))

    assert con_rostro.any()
    assert not sin_rostro.any()
