"""Ventana del smoke test de Fase 0.

Muestra la cámara con el esqueleto de landmarks superpuesto y las métricas que
exigen los criterios de aceptación: FPS de captura, de procesamiento y de
pantalla, latencia, CPU y memoria (para detectar fugas en 10 minutos).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QCloseEvent, QFont, QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from senasperu.config import Config
from senasperu.features.holistic_thread import ProcessedFrame
from senasperu.ui.pipeline_bridge import PipelineBridge, PipelineStats

logger = logging.getLogger(__name__)

# Umbral visual: por debajo de este valor los FPS se muestran en rojo.
FPS_OBJETIVO_MINIMO: float = 25.0
ANCHO_PANEL: int = 250


class SmokeWindow(QMainWindow):
    """Ventana principal del smoke test."""

    def __init__(self, config: Config, *, video_path: str | Path | None = None) -> None:
        """Args:
        config: Configuración cargada.
        video_path: Ruta de un video para probar sin webcam.
        """
        super().__init__()
        self._config = config
        self._bridge = PipelineBridge(config, video_path=video_path, parent=self)

        self.setWindowTitle(
            f"{config.get('proyecto.nombre', 'SeñasPerú')} — Prueba de cámara y landmarks"
        )
        self.resize(960, 600)
        self._build_ui()

        self._bridge.frame_ready.connect(self._on_frame_ready)
        self._bridge.stats_ready.connect(self._on_stats_ready)
        self._bridge.error_occurred.connect(self._on_error)

        QShortcut(QKeySequence(Qt.Key.Key_Escape), self, self.close)

    # -- Construcción de la interfaz --------------------------------------
    def _build_ui(self) -> None:
        central = QWidget(self)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self._video_label = QLabel("Iniciando cámara…")
        self._video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video_label.setMinimumSize(480, 360)
        self._video_label.setStyleSheet("background-color: #101010; color: #d0d0d0;")
        self._video_label.setFrameShape(QFrame.Shape.StyledPanel)
        layout.addWidget(self._video_label, stretch=1)

        layout.addWidget(self._build_panel())
        self.setCentralWidget(central)
        self.statusBar().showMessage("Esc para salir. Ponte a ~1,5 m de la cámara, torso visible.")

    def _build_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(ANCHO_PANEL)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        titulo = QLabel("Rendimiento")
        fuente = titulo.font()
        fuente.setBold(True)
        titulo.setFont(fuente)
        layout.addWidget(titulo)

        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        self._value_labels: dict[str, QLabel] = {}
        campos = [
            ("captura", "Captura"),
            ("proceso", "MediaPipe"),
            ("pantalla", "Pantalla"),
            ("ms", "Tiempo/frame"),
            ("latencia", "Latencia"),
            ("descartados", "Frames descartados"),
            ("manos", "Manos detectadas"),
            ("cpu", "CPU"),
            ("ram", "Memoria"),
            ("tiempo", "Tiempo activo"),
        ]
        for fila, (clave, etiqueta) in enumerate(campos):
            nombre = QLabel(f"{etiqueta}:")
            valor = QLabel("—")
            valor.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            valor.setFont(QFont("Consolas", 10))
            grid.addWidget(nombre, fila, 0)
            grid.addWidget(valor, fila, 1)
            self._value_labels[clave] = valor
        layout.addLayout(grid)

        layout.addSpacing(12)
        self._landmarks_check = QCheckBox("Mostrar landmarks")
        self._landmarks_check.setChecked(bool(self._config.get("ui.mostrar_landmarks", True)))
        self._landmarks_check.toggled.connect(self._bridge.set_draw_landmarks)
        layout.addWidget(self._landmarks_check)

        ayuda = QLabel(
            "Criterios de Fase 0:\n"
            "• ≥25 FPS en pantalla\n"
            "• CPU < 60 %\n"
            "• Memoria estable 10 min"
        )
        ayuda.setWordWrap(True)
        ayuda.setStyleSheet("color: #808080;")
        layout.addWidget(ayuda)
        layout.addStretch(1)
        return panel

    # -- Ciclo de vida -----------------------------------------------------
    def start(self) -> bool:
        """Arranca el pipeline. Devuelve ``False`` si la cámara no se pudo abrir."""
        try:
            self._bridge.start()
        except Exception as error:
            logger.exception("No se pudo iniciar el pipeline")
            self._on_error(str(error))
            return False
        return True

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - API de Qt
        """Detiene los hilos antes de cerrar la ventana."""
        self._bridge.stop()
        super().closeEvent(event)

    # -- Slots -------------------------------------------------------------
    @Slot(object)
    def _on_frame_ready(self, processed: ProcessedFrame) -> None:
        """Pinta el frame más reciente."""
        imagen = _to_qimage(processed.image)
        pixmap = QPixmap.fromImage(imagen).scaled(
            self._video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        self._video_label.setPixmap(pixmap)

    @Slot(object)
    def _on_stats_ready(self, stats: PipelineStats) -> None:
        """Actualiza el panel de métricas."""
        self._set_fps("captura", stats.capture_fps)
        self._set_fps("proceso", stats.process_fps)
        self._set_fps("pantalla", stats.display_fps)
        self._value_labels["ms"].setText(f"{stats.process_ms:.0f} ms")
        self._value_labels["latencia"].setText(f"{stats.latency_ms:.0f} ms")
        self._value_labels["descartados"].setText(str(stats.frames_dropped))
        self._value_labels["manos"].setText(str(stats.hands_detected))
        cpu = "n/d" if np.isnan(stats.cpu_percent) else f"{stats.cpu_percent:.0f} %"
        ram = "n/d" if np.isnan(stats.memory_mb) else f"{stats.memory_mb:.0f} MB"
        self._value_labels["cpu"].setText(cpu)
        self._value_labels["ram"].setText(ram)
        minutos, segundos = divmod(int(stats.elapsed_seconds), 60)
        self._value_labels["tiempo"].setText(f"{minutos:02d}:{segundos:02d}")

    @Slot(str)
    def _on_error(self, mensaje: str) -> None:
        """Muestra el error al usuario en español y deja la app en estado seguro."""
        self._video_label.setText("Sin video")
        QMessageBox.critical(self, "Error de cámara", mensaje)

    def _set_fps(self, clave: str, valor: float) -> None:
        etiqueta = self._value_labels[clave]
        etiqueta.setText(f"{valor:5.1f} FPS")
        color = "#1a7f37" if valor >= FPS_OBJETIVO_MINIMO else "#b42318"
        etiqueta.setStyleSheet(f"color: {color};")


def _to_qimage(image_bgr: np.ndarray) -> QImage:
    """Convierte una imagen BGR de NumPy en ``QImage`` RGB.

    El intercambio de canales se hace con NumPy (no con OpenCV) para que la capa
    de interfaz no dependa de OpenCV.
    """
    alto, ancho = image_bgr.shape[:2]
    rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
    # .copy() desacopla el QImage del buffer de NumPy, que puede liberarse enseguida.
    return QImage(rgb.data, ancho, alto, 3 * ancho, QImage.Format.Format_RGB888).copy()
