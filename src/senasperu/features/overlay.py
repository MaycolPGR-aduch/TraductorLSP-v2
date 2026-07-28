"""Dibujo del esqueleto de landmarks sobre el frame.

Se ejecuta en el hilo de procesamiento (nunca en el hilo de UI) y usa solo
OpenCV/NumPy, de modo que la interfaz Qt jamás toca MediaPipe ni OpenCV.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from senasperu.features.landmarks import POSE_UPPER_BODY, HolisticResult

logger = logging.getLogger(__name__)

# Colores BGR y grosores del esqueleto. Alto contraste para que se distingan
# sobre cualquier ropa o fondo.
COLOR_POSE = (245, 200, 60)          # celeste-azulado
COLOR_LEFT_HAND = (80, 230, 120)     # verde
COLOR_RIGHT_HAND = (80, 140, 250)    # naranja
COLOR_FACE = (200, 200, 200)         # gris claro
GROSOR_LINEA = 2
RADIO_PUNTO = 2
RADIO_PUNTO_MANO = 3
RADIO_PUNTO_ROSTRO = 1


class LandmarkOverlay:
    """Dibuja pose, manos y rostro reducido sobre imágenes BGR."""

    def __init__(self, *, draw_face: bool = True) -> None:
        """Args:
        draw_face: Si se dibujan los puntos del rostro.
        """
        self._draw_face = draw_face
        self._pose_connections: Any | None = None
        self._hand_connections: Any | None = None
        self._load_connections()

    def _load_connections(self) -> None:
        """Carga las listas de conexiones de MediaPipe (aristas del esqueleto)."""
        try:
            import mediapipe as mp

            holistic = mp.solutions.holistic
            self._pose_connections = holistic.POSE_CONNECTIONS
            self._hand_connections = holistic.HAND_CONNECTIONS
        except Exception:  # pragma: no cover - sin MediaPipe solo dibujamos puntos
            logger.debug("No se pudieron cargar las conexiones de MediaPipe; se dibujarán puntos.")

    def draw(self, image: np.ndarray, result: HolisticResult) -> np.ndarray:
        """Dibuja los landmarks sobre la imagen (modifica ``image`` en sitio).

        Args:
            image: Imagen BGR sobre la que dibujar.
            result: Landmarks del frame.

        Returns:
            La misma imagen, ya anotada (se devuelve por comodidad al encadenar).
        """
        alto, ancho = image.shape[:2]

        if result.pose is not None:
            self._draw_part(
                image, result.pose[:, :2], ancho, alto, COLOR_POSE, self._pose_connections,
                RADIO_PUNTO, indices=POSE_UPPER_BODY,
            )
        if self._draw_face and result.face is not None:
            self._draw_points(image, result.face[:, :2], ancho, alto, COLOR_FACE, RADIO_PUNTO_ROSTRO)
        if result.left_hand is not None:
            self._draw_part(
                image, result.left_hand[:, :2], ancho, alto, COLOR_LEFT_HAND,
                self._hand_connections, RADIO_PUNTO_MANO,
            )
        if result.right_hand is not None:
            self._draw_part(
                image, result.right_hand[:, :2], ancho, alto, COLOR_RIGHT_HAND,
                self._hand_connections, RADIO_PUNTO_MANO,
            )
        return image

    def _draw_part(
        self,
        image: np.ndarray,
        puntos_norm: np.ndarray,
        ancho: int,
        alto: int,
        color: tuple[int, int, int],
        conexiones: Any | None,
        radio: int,
        indices: frozenset[int] | None = None,
    ) -> None:
        """Dibuja una parte del cuerpo.

        Args:
            indices: Si se indica, solo se dibujan esos landmarks y las aristas
                cuyos dos extremos estén incluidos.
        """
        # .tolist() entrega enteros de Python, que es lo que espera la API de cv2.
        crudos, dentro = _to_pixels(puntos_norm, ancho, alto)
        pixeles: list[list[int]] = crudos.tolist()
        candidatos = range(len(pixeles)) if indices is None else indices
        visible = {i for i in candidatos if i < len(dentro) and dentro[i]}
        if conexiones:
            total = len(pixeles)
            for inicio, fin in conexiones:
                if inicio in visible and fin in visible and inicio < total and fin < total:
                    cv2.line(
                        image,
                        (pixeles[inicio][0], pixeles[inicio][1]),
                        (pixeles[fin][0], pixeles[fin][1]),
                        color,
                        GROSOR_LINEA,
                        lineType=cv2.LINE_AA,
                    )
        for indice, (x, y) in enumerate(pixeles):
            if indice in visible:
                cv2.circle(image, (x, y), radio, color, -1, lineType=cv2.LINE_AA)

    def _draw_points(
        self,
        image: np.ndarray,
        puntos_norm: np.ndarray,
        ancho: int,
        alto: int,
        color: tuple[int, int, int],
        radio: int,
    ) -> None:
        pixeles, dentro = _to_pixels(puntos_norm, ancho, alto)
        for (x, y), visible in zip(pixeles.tolist(), dentro.tolist()):
            if visible:
                cv2.circle(image, (x, y), radio, color, -1)


def _to_pixels(puntos_norm: np.ndarray, ancho: int, alto: int) -> tuple[np.ndarray, np.ndarray]:
    """Convierte coordenadas normalizadas (0-1) a píxeles enteros.

    MediaPipe extrapola los landmarks que quedan fuera del encuadre, y pegarlos
    al borde dibuja líneas fantasma a lo largo del marco. Por eso se devuelve
    también qué puntos caen realmente dentro de la imagen: los de fuera no se
    dibujan (los datos guardados sí los conservan intactos).

    Returns:
        Tupla ``(pixeles, dentro)``: coordenadas en píxeles y máscara booleana.
    """
    dentro = np.all((puntos_norm >= 0.0) & (puntos_norm <= 1.0), axis=1)
    escalados = puntos_norm * np.array([ancho, alto], dtype=np.float32)
    np.clip(escalados, [0, 0], [ancho - 1, alto - 1], out=escalados)
    return escalados.astype(np.int32), dentro
