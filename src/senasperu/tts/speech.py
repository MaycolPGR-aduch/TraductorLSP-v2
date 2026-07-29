"""Síntesis de voz offline con Piper.

La reproducción ocurre en un hilo aparte: sintetizar una frase toma cientos de
milisegundos y bloquearía la interfaz.

Piper necesita un modelo de voz (``.onnx`` + ``.onnx.json``) que se descarga una
sola vez. Si no está, la app funciona igual y el botón de voz explica qué falta:
la traducción a texto nunca depende de la voz.
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from senasperu.config import Config

logger = logging.getLogger(__name__)

# Espera del hilo por una frase nueva antes de revisar si debe detenerse.
QUEUE_TIMEOUT_SECONDS: float = 0.2

MENSAJE_SIN_MODELO = (
    "No se encontró el modelo de voz de Piper. Descarga '{modelo}.onnx' y "
    "'{modelo}.onnx.json' desde huggingface.co/rhasspy/piper-voices y colócalos "
    "en la carpeta '{carpeta}'. Mientras tanto, la traducción a texto funciona igual."
)


@dataclass(frozen=True, slots=True)
class SpeechStatus:
    """Disponibilidad del motor de voz.

    Attributes:
        available: ``True`` si se puede hablar.
        message: Explicación en español cuando no se puede.
    """

    available: bool
    message: str = ""


class SpeechEngine:
    """Cola de frases habladas con Piper, servida por un hilo trabajador."""

    def __init__(
        self,
        model_path: Path | None,
        *,
        volume: float = 1.0,
        speed: float = 1.0,
        status_message: str = "",
    ) -> None:
        """Args:
        model_path: Ruta del ``.onnx`` de la voz, o ``None`` si no está.
        volume: Volumen de reproducción (0.0-1.0).
        speed: Velocidad del habla; 1.0 es la natural.
        status_message: Motivo por el que la voz no está disponible.
        """
        self._model_path = model_path
        self._volume = float(np.clip(volume, 0.0, 1.0))
        self._speed = float(speed)
        self._status_message = status_message
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._voice = None
        self._failed = False

    @classmethod
    def from_config(cls, config: Config) -> SpeechEngine:
        """Construye el motor con la sección ``tts`` del YAML."""
        nombre = str(config.get("tts.modelo_voz", ""))
        carpeta = config.resolve_path("tts.ruta_modelos_voz")
        modelo = carpeta / f"{nombre}.onnx"
        disponible = modelo.is_file() and modelo.with_suffix(".onnx.json").is_file()
        return cls(
            modelo if disponible else None,
            volume=float(config.get("tts.volumen", 0.9)),
            speed=float(config.get("tts.velocidad", 1.0)),
            status_message=(
                "" if disponible else MENSAJE_SIN_MODELO.format(modelo=nombre, carpeta=carpeta)
            ),
        )

    @property
    def status(self) -> SpeechStatus:
        """Disponibilidad actual del motor."""
        if self._failed:
            return SpeechStatus(False, "El motor de voz falló al iniciarse. Revisa el log.")
        if self._model_path is None:
            return SpeechStatus(False, self._status_message)
        return SpeechStatus(True)

    def start(self) -> None:
        """Arranca el hilo de síntesis, si hay voz disponible."""
        if self._model_path is None or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="voz", daemon=True)
        self._thread.start()

    def speak(self, text: str) -> bool:
        """Encola una frase para hablar.

        Returns:
            ``False`` si no hay voz disponible o el texto está vacío.
        """
        if not text.strip() or self._model_path is None or self._failed:
            return False
        self.start()
        self._queue.put(text)
        return True

    def stop(self, timeout: float = 3.0) -> None:
        """Detiene el hilo de voz."""
        self._stop_event.set()
        self._queue.put(None)
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._thread = None

    # -- Hilo trabajador ---------------------------------------------------
    def _run(self) -> None:
        try:
            self._load_voice()
        except Exception as error:  # pragma: no cover - depende del entorno
            self._failed = True
            logger.exception("No se pudo cargar la voz de Piper: %s", error)
            return

        while not self._stop_event.is_set():
            try:
                texto = self._queue.get(timeout=QUEUE_TIMEOUT_SECONDS)
            except queue.Empty:
                continue
            if texto is None:
                break
            try:
                self._synthesize_and_play(texto)
            except Exception:  # pragma: no cover - fallo de audio en runtime
                logger.exception("Error al reproducir la voz")

    def _load_voice(self) -> None:
        from piper import PiperVoice

        self._voice = PiperVoice.load(str(self._model_path))
        logger.info("Voz de Piper cargada: %s", self._model_path.name)

    def _synthesize_and_play(self, texto: str) -> None:
        import sounddevice as sd
        from piper import SynthesisConfig

        opciones = SynthesisConfig(
            volume=self._volume,
            # En Piper el parámetro es la duración de cada fonema: mayor es más
            # lento, así que es el inverso de la velocidad que pide el YAML.
            length_scale=1.0 / max(0.1, self._speed),
        )
        trozos = list(self._voice.synthesize(texto, syn_config=opciones))
        if not trozos:
            return

        audio = np.concatenate(
            [np.frombuffer(t.audio_int16_bytes, dtype=np.int16) for t in trozos]
        )
        sd.play(audio, samplerate=trozos[0].sample_rate)
        sd.wait()
