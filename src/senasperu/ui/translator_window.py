"""Ventana de la app de traducción en tiempo real (Fase 3).

Pensada para que la lea alguien a un metro de distancia: fuente grande, alto
contraste y el texto acumulándose como una conversación. La interfaz nunca toca
OpenCV ni MediaPipe, y nunca se bloquea: todo llega por señales de Qt.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QCloseEvent, QFont, QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from senasperu.config import Config
from senasperu.features.translation_thread import TranslationFrame
from senasperu.tts.speech import SpeechEngine
from senasperu.ui.translation_bridge import TranslationBridge, TranslationStats
from senasperu.vocabulary import Sign, load_vocabulary

logger = logging.getLogger(__name__)

ANCHO_VIDEO: int = 460

# La frase traducida se pinta con el color de texto del tema, no con un color
# decorativo: es el contenido principal y necesita el máximo contraste posible.
# El acento se reserva para la seña que se está confirmando, que es efímera.
TEMAS = {
    "claro": {
        "fondo": "#ffffff",
        "texto": "#111827",
        "panel": "#f3f4f6",
        "acento": "#065f46",
        "tenue": "#6b7280",
    },
    "oscuro": {
        "fondo": "#111827",
        "texto": "#f9fafb",
        "panel": "#1f2937",
        "acento": "#6ee7b7",
        "tenue": "#9ca3af",
    },
    "alto_contraste": {
        "fondo": "#000000",
        "texto": "#ffff00",
        "panel": "#000000",
        "acento": "#00ffff",
        "tenue": "#ffffff",
    },
}


class TranslatorWindow(QMainWindow):
    """Ventana principal de traducción."""

    def __init__(
        self, config: Config, *, video_path: str | Path | None = None
    ) -> None:
        """Args:
        config: Configuración cargada.
        video_path: Archivo de video en lugar de la webcam (pruebas).
        """
        super().__init__()
        self._config = config
        self._vocabulary: tuple[Sign, ...] = load_vocabulary(config)
        self._bridge = TranslationBridge(config, video_path=video_path, parent=self)
        self._speech = SpeechEngine.from_config(config)
        self._phrase: list[str] = []
        self._max_lines = int(config.get("ui.max_lineas_historial", 50))

        self.setWindowTitle(f"{config.get('proyecto.nombre', 'SeñasPerú')} — Traductor")
        self.resize(1100, 680)
        self._build_ui()
        self._apply_theme()
        self._update_phrase_label()

        self._bridge.frame_ready.connect(self._on_frame_ready)
        self._bridge.stats_ready.connect(self._on_stats_ready)
        self._bridge.error_occurred.connect(self._on_error)

        QShortcut(QKeySequence(Qt.Key.Key_Escape), self, self.close)
        QShortcut(QKeySequence("Ctrl+L"), self, self.clear_conversation)
        QShortcut(QKeySequence("Ctrl+S"), self, self.speak_phrase)

    # -- Construcción ------------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget(self)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        layout.addWidget(self._build_camera_panel())
        layout.addWidget(self._build_text_panel(), stretch=1)
        self.setCentralWidget(central)
        self.statusBar().showMessage(
            "Esc: salir · Ctrl+S: reproducir voz · Ctrl+L: limpiar conversación"
        )

    def _build_camera_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(ANCHO_VIDEO)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._video_label = QLabel("Iniciando cámara…")
        self._video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video_label.setMinimumHeight(340)
        self._video_label.setFrameShape(QFrame.Shape.StyledPanel)
        self._video_label.setStyleSheet("background-color: #101010; color: #d0d0d0;")
        # El video se estira para ocupar el hueco de la columna en vez de dejarlo
        # muerto al fondo.
        layout.addWidget(self._video_label, stretch=1)

        # Seña que se está confirmando: es el feedback de "te estoy entendiendo",
        # así que va grande, justo debajo del video.
        self._candidate_label = QLabel("—")
        self._candidate_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._candidate_label.setFont(QFont("Segoe UI", 26, QFont.Weight.Bold))
        self._candidate_label.setMinimumHeight(46)
        layout.addWidget(self._candidate_label)

        # Indicador de confianza: la barra avanza mientras la seña se confirma.
        self._confidence_bar = QProgressBar()
        self._confidence_bar.setRange(0, 100)
        self._confidence_bar.setTextVisible(True)
        self._confidence_bar.setFormat("Esperando…")
        self._confidence_bar.setMinimumHeight(26)
        layout.addWidget(self._confidence_bar)

        pie = QHBoxLayout()
        self._landmarks_check = QCheckBox("Mostrar landmarks")
        self._landmarks_check.setChecked(bool(self._config.get("ui.mostrar_landmarks", True)))
        self._landmarks_check.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._landmarks_check.toggled.connect(self._bridge.set_draw_landmarks)
        pie.addWidget(self._landmarks_check)
        pie.addStretch(1)

        self._stats_label = QLabel("")
        self._stats_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self._stats_label.setVisible(bool(self._config.get("ui.mostrar_confianza", True)))
        pie.addWidget(self._stats_label)
        layout.addLayout(pie)
        return panel

    def _build_text_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # La frase traducida es el contenido principal de la app: se lleva la
        # mayor parte de la altura y el mayor tamaño de fuente.
        tamano = int(self._config.get("ui.tamano_fuente_traduccion", 32))
        self._phrase_label = QLabel("")
        self._phrase_label.setFont(QFont("Segoe UI", tamano, QFont.Weight.Bold))
        self._phrase_label.setWordWrap(True)
        self._phrase_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        self._phrase_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self._phrase_label, stretch=4)

        botones = QHBoxLayout()
        self._speak_button = QPushButton("Reproducir con voz")
        self._speak_button.clicked.connect(self.speak_phrase)
        self._clear_button = QPushButton("Limpiar")
        self._clear_button.clicked.connect(self.clear_conversation)
        for boton in (self._speak_button, self._clear_button):
            boton.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            boton.setMinimumHeight(44)
            botones.addWidget(boton)
        layout.addLayout(botones)

        # Si falta la voz se avisa en una sola línea discreta: el mensaje
        # completo va en el tooltip y en el diálogo del botón. Un bloque de
        # advertencia a todo color competiría con la traducción, que es lo
        # que el usuario tiene que leer.
        estado_voz = self._speech.status
        self._speak_button.setEnabled(estado_voz.available)
        self._voice_note = QLabel("")
        self._voice_note.setVisible(not estado_voz.available)
        if not estado_voz.available:
            self._speak_button.setText("Voz no disponible")
            self._speak_button.setToolTip(estado_voz.message)
            self._voice_note.setText("Falta el modelo de voz de Piper (ver README).")
            self._voice_note.setToolTip(estado_voz.message)
        layout.addWidget(self._voice_note)

        self._history_title = QLabel("Historial de la conversación")
        layout.addWidget(self._history_title)
        self._history = QTextEdit()
        self._history.setReadOnly(True)
        self._history.setFont(QFont("Segoe UI", 11))
        self._history.setMinimumHeight(120)
        layout.addWidget(self._history, stretch=1)
        return panel

    def _apply_theme(self) -> None:
        """Aplica los colores del tema, garantizando contraste en el texto clave."""
        self._tema = TEMAS.get(str(self._config.get("ui.tema", "claro")), TEMAS["claro"])
        tema = self._tema
        self.setStyleSheet(
            f"QMainWindow, QWidget {{ background-color: {tema['fondo']}; "
            f"color: {tema['texto']}; }} "
            f"QTextEdit {{ background-color: {tema['panel']}; color: {tema['texto']}; "
            f"border: 1px solid {tema['tenue']}; }} "
            f"QPushButton {{ background-color: {tema['panel']}; color: {tema['texto']}; "
            f"border: 1px solid {tema['tenue']}; border-radius: 4px; padding: 6px; }} "
            f"QPushButton:disabled {{ color: {tema['tenue']}; }} "
            f"QProgressBar {{ border: 1px solid {tema['tenue']}; border-radius: 4px; "
            f"text-align: center; color: {tema['texto']}; }} "
            f"QProgressBar::chunk {{ background-color: {tema['acento']}; }}"
        )
        # La frase usa el color de texto del tema: máximo contraste sobre el fondo.
        self._phrase_label.setStyleSheet(f"color: {tema['texto']};")
        self._candidate_label.setStyleSheet(f"color: {tema['acento']};")
        self._stats_label.setStyleSheet(f"color: {tema['tenue']}; font-size: 11px;")
        self._history_title.setStyleSheet(f"color: {tema['tenue']};")
        self._voice_note.setStyleSheet(f"color: {tema['tenue']}; font-size: 11px;")

    # -- Ciclo de vida -----------------------------------------------------
    def start(self) -> bool:
        """Arranca el pipeline. ``False`` si la cámara o el modelo fallan."""
        try:
            self._bridge.start()
        except Exception as error:
            logger.exception("No se pudo iniciar la traducción")
            self._on_error(str(error))
            return False
        return True

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - API de Qt
        """Detiene pipeline y voz antes de cerrar."""
        self._bridge.stop()
        self._speech.stop()
        super().closeEvent(event)

    # -- Acciones ----------------------------------------------------------
    def speak_phrase(self) -> None:
        """Reproduce con voz la frase acumulada."""
        texto = " ".join(self._phrase).strip()
        if not texto:
            return
        if not self._speech.speak(texto):
            QMessageBox.information(self, "Voz no disponible", self._speech.status.message)

    def clear_conversation(self) -> None:
        """Archiva la frase actual en el historial y empieza una nueva."""
        texto = " ".join(self._phrase).strip()
        if texto:
            marca = datetime.now().strftime("%H:%M")
            self._history.append(f"[{marca}] {texto}")
            self._trim_history()
        self._phrase.clear()
        self._update_phrase_label()

    @property
    def current_phrase(self) -> str:
        """Frase acumulada en pantalla."""
        return " ".join(self._phrase).strip()

    # -- Slots -------------------------------------------------------------
    @Slot(object)
    def _on_frame_ready(self, frame: TranslationFrame) -> None:
        """Pinta el video y actualiza la traducción."""
        rgb = np.ascontiguousarray(frame.image[:, :, ::-1])
        alto, ancho = rgb.shape[:2]
        imagen = QImage(rgb.data, ancho, alto, 3 * ancho, QImage.Format.Format_RGB888)
        self._video_label.setPixmap(
            QPixmap.fromImage(imagen).scaled(
                self._video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
        )

        if frame.state is None:
            porcentaje = int(100 * frame.buffer_ratio)
            self._confidence_bar.setValue(porcentaje)
            self._confidence_bar.setFormat(f"Llenando la ventana… {porcentaje}%")
            return

        estado = frame.state
        self._confidence_bar.setValue(int(100 * estado.progress))
        if estado.at_rest:
            self._confidence_bar.setFormat("En reposo")
            self._candidate_label.setText("—")
        elif estado.candidate is not None:
            sign = self._vocabulary[estado.candidate]
            self._confidence_bar.setFormat(f"{estado.candidate_confidence:.0%} de confianza")
            self._candidate_label.setText(sign.glosa)
        else:
            self._confidence_bar.setFormat("Sin señal clara")
            self._candidate_label.setText("—")

        if estado.confirmed is not None:
            self._append_sign(self._vocabulary[estado.confirmed])

    @Slot(object)
    def _on_stats_ready(self, stats: TranslationStats) -> None:
        """Actualiza la línea de métricas."""
        self._stats_label.setText(
            f"{stats.display_fps:.0f} FPS · MediaPipe {stats.process_ms:.0f} ms · "
            f"modelo {stats.inference_ms:.1f} ms · latencia {stats.latency_ms:.0f} ms"
        )

    @Slot(str)
    def _on_error(self, mensaje: str) -> None:
        """Muestra el error y deja la app en estado seguro."""
        self._video_label.setText("Sin video")
        self._confidence_bar.setFormat("Detenido")
        QMessageBox.critical(self, "Error", mensaje)

    # -- Interno -----------------------------------------------------------
    def _append_sign(self, sign: Sign) -> None:
        """Agrega una seña confirmada a la frase en pantalla."""
        if not sign.text:
            return
        self._phrase.append(sign.text)
        self._update_phrase_label()
        if bool(self._config.get("tts.reproducir_automatico", False)):
            self._speech.speak(sign.text)

    def _update_phrase_label(self) -> None:
        """Muestra la frase, o un texto guía cuando todavía no hay nada.

        Un área grande y vacía se lee como "la app no funciona"; el texto guía
        deja claro que está esperando.
        """
        texto = " ".join(self._phrase).strip()
        if texto:
            self._phrase_label.setText(texto)
            self._phrase_label.setStyleSheet(f"color: {self._tema['texto']};")
        else:
            self._phrase_label.setText("Ponte frente a la cámara y empieza a señar…")
            self._phrase_label.setStyleSheet(f"color: {self._tema['tenue']};")

    def _trim_history(self) -> None:
        """Recorta el historial para que no crezca sin límite."""
        lineas = self._history.toPlainText().splitlines()
        if len(lineas) > self._max_lines:
            self._history.setPlainText("\n".join(lineas[-self._max_lines :]))
