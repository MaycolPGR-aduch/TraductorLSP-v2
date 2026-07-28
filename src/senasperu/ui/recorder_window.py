"""Grabador de dataset (Fase 1).

Diseñado para que grabar 15 repeticiones seguidas sea rápido y no requiera el
mouse, y para que **etiquetar mal sea imposible**: la seña se elige de la lista
del vocabulario, se congela al iniciar la cuenta regresiva y el nombre del
archivo se deriva de ese estado. No hay ningún campo de texto libre.

Ciclo de una repetición:
``Espacio`` → cuenta regresiva → captura → veredicto de calidad → guardado
(en un hilo aparte, para que la vista previa nunca se corte).
"""

from __future__ import annotations

import logging
import math
import time
from enum import Enum, auto
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QCloseEvent, QFont, QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from senasperu.config import Config
from senasperu.data.dataset_writer import DatasetWriter, SavedRecording
from senasperu.data.quality import QualityChecker
from senasperu.data.recording import RecordingBuffer
from senasperu.data.save_worker import DatasetSaveWorker
from senasperu.features.holistic_thread import ProcessedFrame
from senasperu.features.vector import layout_from_config
from senasperu.ui.pipeline_bridge import PipelineBridge
from senasperu.vocabulary import Sign, load_vocabulary

logger = logging.getLogger(__name__)

# Cada cuántos milisegundos avanza la máquina de estados de la grabación.
STATE_TICK_MS: int = 50
ANCHO_PANEL: int = 330

COLOR_OK = "#1a7f37"
COLOR_ERROR = "#b42318"
COLOR_AVISO = "#b54708"
COLOR_NEUTRO = "#404040"


class RecorderState(Enum):
    """Estados del grabador."""

    IDLE = auto()
    COUNTDOWN = auto()
    RECORDING = auto()


class RecorderWindow(QMainWindow):
    """Ventana del grabador de dataset."""

    def __init__(
        self,
        config: Config,
        *,
        person: str | None = None,
        video_path: str | Path | None = None,
    ) -> None:
        """Args:
        config: Configuración cargada.
        person: Identificador de la persona señante (``p01``). Si es ``None``,
            se toma de ``dataset.persona_actual``.
        video_path: Archivo de video en lugar de la webcam (pruebas).
        """
        super().__init__()
        self._config = config
        self._person = person or str(config.require("dataset.persona_actual"))
        self._vocabulary: tuple[Sign, ...] = load_vocabulary(config)
        self._current_index = 0

        self._countdown_seconds = float(config.require("grabador.cuenta_regresiva_segundos"))
        self._recording_seconds = float(config.require("grabador.duracion_grabacion_segundos"))
        self._target_per_sign = int(config.require("grabador.repeticiones_objetivo_por_sena"))
        self._max_per_session = int(config.require("grabador.repeticiones_max_por_sesion"))

        self._writer = DatasetWriter.from_config(config)
        self._worker = DatasetSaveWorker(self._writer)
        self._checker = QualityChecker.from_config(config)
        self._buffer = RecordingBuffer(
            layout_from_config(config),
            keep_video=bool(config.get("grabador.guardar_video_respaldo", False)),
        )

        self._state = RecorderState.IDLE
        self._deadline = 0.0
        self._recording_sign: Sign | None = None
        self._last_saved: SavedRecording | None = None
        self._last_sign_id: str | None = None
        self._rejected_in_session = 0

        self._bridge = PipelineBridge(config, video_path=video_path, parent=self)

        self.setWindowTitle(
            f"{config.get('proyecto.nombre', 'SeñasPerú')} — Grabador de dataset"
        )
        self.resize(1120, 680)
        self._build_ui()
        self._session_spin.setValue(self._writer.next_session(self._person))
        self._refresh_counters()

        self._bridge.frame_ready.connect(self._on_frame_ready)
        self._bridge.error_occurred.connect(self._on_error)

        self._state_timer = QTimer(self)
        self._state_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._state_timer.setInterval(STATE_TICK_MS)
        self._state_timer.timeout.connect(self._tick)

        self._install_shortcuts()

    # -- Construcción de la interfaz --------------------------------------
    def _build_ui(self) -> None:
        central = QWidget(self)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        layout.addWidget(self._build_video_area(), stretch=1)
        layout.addWidget(self._build_panel())
        self.setCentralWidget(central)
        self.statusBar().showMessage(self._shortcut_help())

    def _build_video_area(self) -> QWidget:
        contenedor = QWidget()
        rejilla = QGridLayout(contenedor)
        rejilla.setContentsMargins(0, 0, 0, 0)
        rejilla.setSpacing(0)

        self._video_label = QLabel("Iniciando cámara…")
        self._video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video_label.setMinimumSize(560, 420)
        self._video_label.setFrameShape(QFrame.Shape.Box)
        self._video_label.setLineWidth(3)
        self._set_video_border(COLOR_NEUTRO)

        # El cartel grande va apilado sobre el video, en la misma celda.
        self._overlay_label = QLabel("")
        self._overlay_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._overlay_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._overlay_label.setFont(QFont("Segoe UI", 96, QFont.Weight.Bold))
        self._overlay_label.setStyleSheet("color: #ffd166; background: transparent;")

        rejilla.addWidget(self._video_label, 0, 0)
        rejilla.addWidget(self._overlay_label, 0, 0)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setTextVisible(False)
        self._progress.setFixedHeight(10)
        rejilla.addWidget(self._progress, 1, 0)
        return contenedor

    def _build_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(ANCHO_PANEL)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Identificación de la sesión
        cabecera = QGridLayout()
        cabecera.addWidget(QLabel("Persona:"), 0, 0)
        etiqueta_persona = QLabel(self._person)
        etiqueta_persona.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
        cabecera.addWidget(etiqueta_persona, 0, 1)
        cabecera.addWidget(QLabel("Sesión:"), 1, 0)
        self._session_spin = QSpinBox()
        self._session_spin.setRange(1, 99)
        self._session_spin.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._session_spin.valueChanged.connect(self._refresh_counters)
        cabecera.addWidget(self._session_spin, 1, 1)
        layout.addLayout(cabecera)

        # Condiciones de grabación
        self._condition_boxes: dict[str, QComboBox] = {}
        condiciones = QGridLayout()
        for fila, clave in enumerate(("iluminacion", "distancia", "ropa")):
            opciones = list(self._config.get(f"grabador.condiciones.{clave}", []) or [])
            if not opciones:
                continue
            combo = QComboBox()
            combo.addItems(opciones)
            combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            condiciones.addWidget(QLabel(f"{clave.capitalize()}:"), fila, 0)
            condiciones.addWidget(combo, fila, 1)
            self._condition_boxes[clave] = combo
        layout.addLayout(condiciones)

        # Seña actual, en grande
        self._glosa_label = QLabel("—")
        self._glosa_label.setFont(QFont("Segoe UI", 26, QFont.Weight.Bold))
        self._glosa_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._glosa_label.setWordWrap(True)
        layout.addSpacing(6)
        layout.addWidget(self._glosa_label)

        self._counter_label = QLabel("")
        self._counter_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._counter_label)

        self._status_label = QLabel("Listo para grabar.")
        self._status_label.setWordWrap(True)
        self._status_label.setMinimumHeight(56)
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self._status_label)

        # Botones (sin foco: los atajos de teclado mandan)
        botones = QHBoxLayout()
        self._record_button = QPushButton("Grabar")
        self._record_button.clicked.connect(self.start_recording)
        self._discard_button = QPushButton("Descartar última")
        self._discard_button.clicked.connect(self.discard_last)
        self._discard_button.setEnabled(False)
        for boton in (self._record_button, self._discard_button):
            boton.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            botones.addWidget(boton)
        layout.addLayout(botones)

        # Vocabulario con progreso
        self._sign_list = QListWidget()
        self._sign_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._sign_list.setFont(QFont("Consolas", 9))
        for sign in self._vocabulary:
            self._sign_list.addItem(QListWidgetItem(sign.glosa))
        self._sign_list.currentRowChanged.connect(self._on_row_changed)
        layout.addWidget(self._sign_list, stretch=1)

        ayuda = QLabel(self._shortcut_help())
        ayuda.setWordWrap(True)
        ayuda.setStyleSheet("color: #808080;")
        layout.addWidget(ayuda)
        return panel

    def _install_shortcuts(self) -> None:
        """Conecta los atajos de teclado definidos en la configuración."""
        acciones = {
            "grabar": self.start_recording,
            "repetir_ultima": self.repeat_last,
            "descartar_ultima": self.discard_last,
            "siguiente_sena": lambda: self._move_selection(1),
            "anterior_sena": lambda: self._move_selection(-1),
        }
        for clave, accion in acciones.items():
            tecla = self._config.get(f"grabador.atajos.{clave}")
            if not tecla:
                continue
            atajo = QShortcut(QKeySequence(str(tecla)), self)
            atajo.setContext(Qt.ShortcutContext.ApplicationShortcut)
            atajo.activated.connect(accion)
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self, self.close)

    def _shortcut_help(self) -> str:
        atajos = self._config.get("grabador.atajos", {})
        return (
            f"{atajos.get('grabar', 'Space')}: grabar · "
            f"{atajos.get('repetir_ultima', 'R')}: repetir última · "
            f"{atajos.get('descartar_ultima', 'D')}: descartar última · "
            f"{atajos.get('anterior_sena', 'Left')}/{atajos.get('siguiente_sena', 'Right')}: "
            "cambiar seña · Esc: salir"
        )

    # -- Ciclo de vida -----------------------------------------------------
    def start(self) -> bool:
        """Arranca cámara, hilo de escritura y máquina de estados."""
        try:
            self._bridge.start()
        except Exception as error:
            logger.exception("No se pudo iniciar la cámara del grabador")
            self._on_error(str(error))
            return False
        self._worker.start()
        self._state_timer.start()
        self._sign_list.setCurrentRow(0)
        return True

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - API de Qt
        """Detiene todo, esperando a que se escriba lo pendiente."""
        self._state_timer.stop()
        self._bridge.stop()
        if self._worker.is_alive():
            if self._worker.pending:
                self._status_label.setText("Guardando lo pendiente…")
            self._worker.stop()
        super().closeEvent(event)

    # -- Acciones ----------------------------------------------------------
    @property
    def current_sign(self) -> Sign:
        """Seña seleccionada en este momento."""
        return self._vocabulary[self._current_index]

    def start_recording(self) -> None:
        """Inicia la cuenta regresiva de una nueva repetición."""
        if self._state is not RecorderState.IDLE:
            return
        if not self._bridge.is_running:
            self._set_status("La cámara no está activa.", COLOR_ERROR)
            return
        self._recording_sign = self.current_sign
        self._state = RecorderState.COUNTDOWN
        self._deadline = time.perf_counter() + self._countdown_seconds
        self._set_video_border(COLOR_AVISO)
        self._set_status(f"Prepárate para {self._recording_sign.glosa}…", COLOR_AVISO)
        self._update_controls()

    def repeat_last(self) -> None:
        """Vuelve a grabar la última seña grabada."""
        if self._state is not RecorderState.IDLE:
            return
        if self._last_sign_id is not None:
            for indice, sign in enumerate(self._vocabulary):
                if sign.id == self._last_sign_id:
                    self._sign_list.setCurrentRow(indice)
                    break
        self.start_recording()

    def discard_last(self) -> None:
        """Borra la última repetición guardada (deshacer de una sola vez)."""
        if self._state is not RecorderState.IDLE or self._last_saved is None:
            return
        descartada = self._last_saved
        self._last_saved = None
        self._discard_button.setEnabled(False)
        self._worker.submit_discard(descartada)
        self._set_status(f"Descartando {descartada.stem}…", COLOR_AVISO)

    def _move_selection(self, delta: int) -> None:
        """Cambia de seña. Se ignora durante una grabación, para no mal etiquetar."""
        if self._state is not RecorderState.IDLE:
            return
        total = len(self._vocabulary)
        self._sign_list.setCurrentRow((self._current_index + delta) % total)

    @Slot(int)
    def _on_row_changed(self, row: int) -> None:
        if 0 <= row < len(self._vocabulary):
            self._current_index = row
            self._refresh_counters()

    # -- Máquina de estados ------------------------------------------------
    def _tick(self) -> None:
        """Avanza cuenta regresiva y grabación, y recoge resultados de escritura."""
        ahora = time.perf_counter()
        restante = self._deadline - ahora

        if self._state is RecorderState.COUNTDOWN:
            if restante <= 0:
                self._begin_capture()
            else:
                self._overlay_label.setText(str(max(1, math.ceil(restante))))
        elif self._state is RecorderState.RECORDING:
            transcurrido = self._recording_seconds - max(0.0, restante)
            self._progress.setValue(
                int(100 * min(1.0, transcurrido / max(0.001, self._recording_seconds)))
            )
            if restante <= 0:
                self._finish_capture()

        self._collect_save_results()

    def _begin_capture(self) -> None:
        """Pasa de la cuenta regresiva a la captura efectiva."""
        sign = self._recording_sign or self.current_sign
        self._buffer.start(sign.id)
        self._state = RecorderState.RECORDING
        self._deadline = time.perf_counter() + self._recording_seconds
        self._overlay_label.setText("●")
        self._overlay_label.setStyleSheet(f"color: {COLOR_ERROR}; background: transparent;")
        self._set_video_border(COLOR_ERROR)
        self._set_status(f"Grabando {sign.glosa}…", COLOR_ERROR)

    def _finish_capture(self) -> None:
        """Cierra la grabación, la evalúa y la manda a guardar si pasa el control."""
        sign = self._recording_sign or self.current_sign
        self._state = RecorderState.IDLE
        self._recording_sign = None
        self._overlay_label.setText("")
        self._overlay_label.setStyleSheet("color: #ffd166; background: transparent;")
        self._progress.setValue(0)
        self._set_video_border(COLOR_NEUTRO)

        muestra = self._buffer.build()
        informe = self._checker.evaluate(muestra.hands_per_frame, muestra.confidence)
        self._last_sign_id = sign.id

        if not informe.accepted:
            self._rejected_in_session += 1
            logger.warning("Grabación rechazada de %s: %s", sign.glosa, informe.summary)
            self._set_status(
                f"RECHAZADA — {' '.join(informe.reasons)}\nVuelve a grabar con "
                f"{self._config.get('grabador.atajos.repetir_ultima', 'R')}.",
                COLOR_ERROR,
            )
            self._update_controls()
            return

        self._worker.submit(
            muestra,
            person=self._person,
            session=int(self._session_spin.value()),
            report=informe,
            conditions={
                clave: combo.currentText() for clave, combo in self._condition_boxes.items()
            },
        )
        self._set_status(f"Aceptada — {informe.summary}. Guardando…", COLOR_OK)
        self._update_controls()

    def _collect_save_results(self) -> None:
        """Recoge lo que haya terminado el hilo de escritura."""
        while (resultado := self._worker.poll()) is not None:
            if resultado.error:
                self._set_status(resultado.error, COLOR_ERROR)
            elif resultado.saved is not None:
                self._last_saved = resultado.saved
                self._discard_button.setEnabled(True)
                self._set_status(f"Guardada {resultado.saved.stem}", COLOR_OK)
            elif resultado.discarded is not None:
                self._set_status(f"Descartada {resultado.discarded.stem}", COLOR_AVISO)
            self._refresh_counters()

    # -- Slots del pipeline ------------------------------------------------
    @Slot(object)
    def _on_frame_ready(self, processed: ProcessedFrame) -> None:
        """Pinta el frame y, si se está grabando, lo acumula en el buffer."""
        if self._state is RecorderState.RECORDING:
            self._buffer.add(
                processed.result,
                processed.capture_timestamp,
                processed.clean_image,
            )

        rgb = np.ascontiguousarray(processed.image[:, :, ::-1])
        alto, ancho = rgb.shape[:2]
        imagen = QImage(rgb.data, ancho, alto, 3 * ancho, QImage.Format.Format_RGB888)
        self._video_label.setPixmap(
            QPixmap.fromImage(imagen).scaled(
                self._video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
        )

    @Slot(str)
    def _on_error(self, mensaje: str) -> None:
        """Muestra el error de cámara y deja el grabador en estado seguro."""
        self._state = RecorderState.IDLE
        self._video_label.setText("Sin video")
        self._set_status(mensaje, COLOR_ERROR)
        QMessageBox.critical(self, "Error de cámara", mensaje)

    # -- Interfaz ----------------------------------------------------------
    def _refresh_counters(self) -> None:
        """Actualiza la seña activa, sus contadores y la lista de progreso."""
        sesion = int(self._session_spin.value())
        sign = self.current_sign
        en_sesion = self._writer.count(sign.id, self._person, sesion)
        total = self._writer.count(sign.id, self._person)

        self._glosa_label.setText(sign.glosa)
        detalle = f"Sesión {sesion:02d}: {en_sesion}/{self._max_per_session}"
        detalle += f"   ·   Total: {total}/{self._target_per_sign}"
        if self._rejected_in_session:
            detalle += f"\nRechazadas en esta sesión: {self._rejected_in_session}"
        self._counter_label.setText(detalle)

        totales = self._writer.counts_by_label(self._person)
        for fila, entrada in enumerate(self._vocabulary):
            cantidad = totales.get(entrada.id, 0)
            item = self._sign_list.item(fila)
            if item is not None:
                item.setText(f"{entrada.glosa:<14} {cantidad:>3}/{self._target_per_sign}")

        if en_sesion >= self._max_per_session and self._state is RecorderState.IDLE:
            self._set_status(
                f"Ya tienes {en_sesion} repeticiones de {sign.glosa} en esta sesión. "
                "Cambia de seña o abre una sesión nueva: repetir tanto seguido produce "
                "muestras casi idénticas.",
                COLOR_AVISO,
            )

    def _update_controls(self) -> None:
        libre = self._state is RecorderState.IDLE
        self._record_button.setEnabled(libre)
        self._session_spin.setEnabled(libre)

    def _set_status(self, mensaje: str, color: str = COLOR_NEUTRO) -> None:
        self._status_label.setText(mensaje)
        self._status_label.setStyleSheet(f"color: {color};")

    def _set_video_border(self, color: str) -> None:
        self._video_label.setStyleSheet(
            f"background-color: #101010; color: #d0d0d0; border: 3px solid {color};"
        )
