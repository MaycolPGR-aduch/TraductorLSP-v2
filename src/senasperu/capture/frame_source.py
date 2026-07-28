"""Fuentes de frames: webcam real y archivo de video (para pruebas sin cámara).

La aplicación depende de la interfaz :class:`FrameSource`, nunca de OpenCV
directamente. Así cualquier módulo puede probarse sin webcam inyectando
:class:`VideoFileSource`.
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import cv2
import numpy as np

from senasperu.config import Config

logger = logging.getLogger(__name__)


class CameraError(RuntimeError):
    """No se pudo abrir o leer la cámara/archivo de video."""


@dataclass(frozen=True, slots=True)
class Frame:
    """Un frame capturado, ya listo para procesar.

    Attributes:
        image: Imagen BGR (formato nativo de OpenCV), ya espejada si corresponde.
        index: Número correlativo del frame desde que se abrió la fuente.
        timestamp: Marca de tiempo monótona (``time.perf_counter``) de la captura.
    """

    image: np.ndarray
    index: int
    timestamp: float


@runtime_checkable
class FrameSource(Protocol):
    """Interfaz mínima de una fuente de frames."""

    def open(self) -> None:
        """Abre el dispositivo o archivo. Lanza :class:`CameraError` si falla."""
        ...

    def read(self) -> Frame | None:
        """Devuelve el siguiente frame, o ``None`` si la fuente terminó o falló."""
        ...

    def close(self) -> None:
        """Libera los recursos. Debe ser idempotente."""
        ...

    @property
    def is_open(self) -> bool:
        """``True`` si la fuente está lista para entregar frames."""
        ...

    @property
    def description(self) -> str:
        """Texto corto para mostrar al usuario o registrar en el log."""
        ...


class _BaseSource:
    """Estado común de las fuentes basadas en ``cv2.VideoCapture``."""

    def __init__(self, *, mirror: bool) -> None:
        self._capture: cv2.VideoCapture | None = None
        self._mirror = mirror
        self._index = 0

    @property
    def is_open(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self):  # noqa: D105 - azúcar de contexto
        self.open()
        return self

    def __exit__(self, *_exc_info) -> None:  # noqa: D105
        self.close()

    def _build_frame(self, image: np.ndarray) -> Frame:
        if self._mirror:
            # Espejamos en la captura (no solo en pantalla) para que lo que el
            # usuario ve, lo que MediaPipe procesa y lo que se graba coincidan.
            image = cv2.flip(image, 1)
        frame = Frame(image=image, index=self._index, timestamp=time.perf_counter())
        self._index += 1
        return frame

    def open(self) -> None:  # pragma: no cover - implementado por subclases
        raise NotImplementedError


class CameraSource(_BaseSource):
    """Webcam vía OpenCV, configurada desde ``config/default.yaml``."""

    def __init__(
        self,
        device_index: int,
        width: int,
        height: int,
        fps: int,
        *,
        mirror: bool = True,
        backend: int | None = None,
    ) -> None:
        """Args:
        device_index: Índice del dispositivo (0 = webcam por defecto).
        width: Ancho solicitado en píxeles.
        height: Alto solicitado en píxeles.
        fps: FPS solicitados al driver.
        mirror: Si se espeja horizontalmente cada frame.
        backend: Backend de OpenCV. Si es ``None``, se usa DirectShow en
            Windows (apertura mucho más rápida) y el automático en otros SO.
        """
        super().__init__(mirror=mirror)
        self._device_index = device_index
        self._width = width
        self._height = height
        self._fps = fps
        self._backend = backend if backend is not None else _default_backend()

    @classmethod
    def from_config(cls, config: Config) -> CameraSource:
        """Construye la fuente a partir de la sección ``camara`` del YAML."""
        return cls(
            device_index=int(config.require("camara.indice")),
            width=int(config.require("camara.ancho")),
            height=int(config.require("camara.alto")),
            fps=int(config.require("camara.fps_objetivo")),
            mirror=bool(config.get("camara.espejar", True)),
        )

    def open(self) -> None:
        """Abre la webcam y aplica resolución y FPS solicitados."""
        if self.is_open:
            return
        capture = cv2.VideoCapture(self._device_index, self._backend)
        if not capture.isOpened():
            capture.release()
            raise CameraError(
                f"No se pudo abrir la cámara con índice {self._device_index}. "
                "Verifica que esté conectada, que ninguna otra aplicación la esté usando "
                "y que Windows tenga permiso de cámara habilitado."
            )
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        capture.set(cv2.CAP_PROP_FPS, self._fps)
        # Buffer mínimo: si el driver lo respeta, evita entregar frames rancios.
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._capture = capture
        self._index = 0
        logger.info(
            "Cámara %s abierta a %sx%s (solicitado %sx%s @ %s FPS)",
            self._device_index,
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            self._width,
            self._height,
            self._fps,
        )

    def read(self) -> Frame | None:
        """Lee un frame de la webcam; ``None`` si la lectura falla."""
        if self._capture is None:
            return None
        ok, image = self._capture.read()
        if not ok or image is None:
            return None
        return self._build_frame(image)

    @property
    def description(self) -> str:
        return f"Cámara {self._device_index} ({self._width}x{self._height})"


class VideoFileSource(_BaseSource):
    """Archivo de video como fuente de frames. Permite probar sin webcam."""

    def __init__(
        self,
        path: str | Path,
        *,
        mirror: bool = False,
        loop: bool = False,
        realtime: bool = False,
    ) -> None:
        """Args:
        path: Ruta del archivo de video.
        mirror: Si se espeja horizontalmente cada frame.
        loop: Si al llegar al final vuelve a empezar.
        realtime: Si se respeta la cadencia original del video (útil para
            simular una cámara en pruebas de rendimiento).
        """
        super().__init__(mirror=mirror)
        self._path = Path(path)
        self._loop = loop
        self._realtime = realtime
        self._frame_interval = 0.0
        self._next_deadline = 0.0

    def open(self) -> None:
        """Abre el archivo de video."""
        if self.is_open:
            return
        if not self._path.is_file():
            raise CameraError(f"No se encontró el archivo de video: {self._path}")
        capture = cv2.VideoCapture(str(self._path))
        if not capture.isOpened():
            capture.release()
            raise CameraError(f"No se pudo abrir el archivo de video: {self._path}")
        self._capture = capture
        self._index = 0
        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        self._frame_interval = 1.0 / fps if (self._realtime and fps > 0) else 0.0
        self._next_deadline = time.perf_counter()
        logger.info("Archivo de video abierto: %s (%.1f FPS)", self._path, fps)

    def read(self) -> Frame | None:
        """Lee el siguiente frame del archivo; ``None`` al terminar (si no hay bucle)."""
        if self._capture is None:
            return None
        if self._frame_interval > 0:
            espera = self._next_deadline - time.perf_counter()
            if espera > 0:
                time.sleep(espera)
            self._next_deadline = max(
                self._next_deadline + self._frame_interval, time.perf_counter()
            )
        ok, image = self._capture.read()
        if not ok or image is None:
            if not self._loop:
                return None
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, image = self._capture.read()
            if not ok or image is None:
                return None
        return self._build_frame(image)

    @property
    def description(self) -> str:
        return f"Video {self._path.name}"


def _default_backend() -> int:
    """Backend de OpenCV recomendado para la plataforma actual."""
    if sys.platform.startswith("win"):
        return cv2.CAP_DSHOW
    return cv2.CAP_ANY


def create_frame_source(config: Config, video_path: str | Path | None = None) -> FrameSource:
    """Fábrica de fuentes: archivo de video si se indica una ruta, si no la webcam.

    Args:
        config: Configuración cargada.
        video_path: Ruta a un video para pruebas sin cámara.
    """
    if video_path is not None:
        return VideoFileSource(
            video_path,
            mirror=bool(config.get("camara.espejar", True)),
            loop=True,
            realtime=True,
        )
    return CameraSource.from_config(config)
