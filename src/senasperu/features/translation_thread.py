"""Hilo de inferencia de la app de traducción (Fase 3).

Encadena todo lo que hay entre la cámara y la interfaz:

``frames → MediaPipe → normalización → ventana deslizante → ONNX → estabilización``

Vive en un hilo propio para que la interfaz nunca se bloquee, y publica sus
resultados en una cola con descarte del más viejo: si la UI se atrasa, preferimos
perder un fotograma antes que acumular latencia.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import numpy as np

from senasperu.capture.frame_queue import DropOldestQueue
from senasperu.capture.frame_source import Frame
from senasperu.config import Config
from senasperu.features.holistic import HolisticExtractor
from senasperu.features.landmarks import HolisticResult
from senasperu.features.normalize import LandmarkNormalizer
from senasperu.features.overlay import LandmarkOverlay
from senasperu.features.stream_buffer import StreamWindowBuffer
from senasperu.features.vector import layout_from_config
from senasperu.model.inference import Prediction, SignClassifier
from senasperu.stabilize.stabilizer import Stabilizer, StabilizerState
from senasperu.utils import FpsMeter, LatencyMeter
from senasperu.vocabulary import REST_SIGN_ID, Sign, find_sign, load_vocabulary

logger = logging.getLogger(__name__)

QUEUE_TIMEOUT_SECONDS: float = 0.2


@dataclass(frozen=True, slots=True)
class TranslationFrame:
    """Lo que la interfaz necesita para pintar un fotograma.

    Attributes:
        image: Imagen BGR con el esqueleto dibujado.
        result: Landmarks del frame.
        state: Estado del estabilizador, o ``None`` si aún no se infirió.
        prediction: Predicción cruda del modelo, para el indicador de confianza.
        buffer_ratio: Cuánto lleva llena la ventana inicial (0.0-1.0).
        latency_seconds: Tiempo desde la captura hasta este resultado.
    """

    image: np.ndarray
    result: HolisticResult
    state: StabilizerState | None
    prediction: Prediction | None
    buffer_ratio: float
    latency_seconds: float


class TranslationThread(threading.Thread):
    """Hilo de MediaPipe + modelo + estabilización."""

    def __init__(
        self,
        config: Config,
        frame_queue: DropOldestQueue[Frame],
        result_queue: DropOldestQueue[TranslationFrame],
        *,
        classifier: SignClassifier | None = None,
        name: str = "inferencia",
    ) -> None:
        """Args:
        config: Configuración cargada.
        frame_queue: Cola de entrada del hilo de captura.
        result_queue: Cola de salida hacia la interfaz.
        classifier: Clasificador ya cargado. Si es ``None``, se carga desde la
            configuración dentro del hilo.
        name: Nombre del hilo.
        """
        super().__init__(name=name, daemon=True)
        self._config = config
        self._frame_queue = frame_queue
        self._result_queue = result_queue
        self._classifier = classifier
        self._stop_event = threading.Event()

        self._vocabulary: tuple[Sign, ...] = load_vocabulary(config)
        reposo = find_sign(self._vocabulary, REST_SIGN_ID)
        self._rest_index = reposo.index if reposo is not None else 0

        self._fps_meter = FpsMeter()
        self._process_meter = LatencyMeter()
        self._inference_meter = LatencyMeter()
        self._error: str | None = None
        self._frames_processed = 0
        self._inferences = 0
        self.draw_landmarks: bool = bool(config.get("ui.mostrar_landmarks", True))

    def run(self) -> None:  # noqa: D102 - documentado en la clase
        logger.info("Hilo de inferencia iniciado")
        extractor: HolisticExtractor | None = None
        try:
            extractor = HolisticExtractor.from_config(self._config)
            overlay = LandmarkOverlay(
                draw_face=bool(self._config.get("mediapipe.usar_rostro", True))
            )
            layout = layout_from_config(self._config)
            aspecto = float(self._config.require("camara.ancho")) / float(
                self._config.require("camara.alto")
            )
            normalizador = LandmarkNormalizer.from_config(
                self._config, layout, aspect_ratio=aspecto
            )
            buffer = StreamWindowBuffer.from_config(self._config, layout, normalizador)
            estabilizador = Stabilizer.from_config(self._config, self._rest_index)
            clasificador = self._classifier or SignClassifier.from_config(self._config)

            while not self._stop_event.is_set():
                frame = self._frame_queue.get(timeout=QUEUE_TIMEOUT_SECONDS)
                if frame is None:
                    continue

                inicio = time.perf_counter()
                resultado = extractor.process(frame.image)
                tras_mediapipe = time.perf_counter()

                ventana = buffer.push(resultado)
                prediccion: Prediction | None = None
                estado: StabilizerState | None = None
                if ventana is not None:
                    prediccion = clasificador.predict(ventana)
                    estado = estabilizador.update(
                        prediccion.class_index, prediccion.confidence, tras_mediapipe
                    )
                    self._inferences += 1
                    self._inference_meter.add(time.perf_counter() - tras_mediapipe)

                imagen = frame.image
                if self.draw_landmarks:
                    imagen = overlay.draw(frame.image.copy(), resultado)

                ahora = time.perf_counter()
                self._frames_processed += 1
                self._fps_meter.tick(ahora)
                self._process_meter.add(tras_mediapipe - inicio)

                self._result_queue.put(
                    TranslationFrame(
                        image=imagen,
                        result=resultado,
                        state=estado,
                        prediction=prediccion,
                        buffer_ratio=buffer.ready_ratio,
                        latency_seconds=ahora - frame.timestamp,
                    )
                )
        except FileNotFoundError as error:
            self._error = str(error)
            logger.error("%s", error)
        except Exception as error:  # pragma: no cover - red de seguridad
            self._error = f"Error en la inferencia: {error}"
            logger.exception("Error en el hilo de inferencia")
        finally:
            if extractor is not None:
                extractor.close()
            logger.info(
                "Hilo de inferencia detenido: %s frames, %s inferencias",
                self._frames_processed,
                self._inferences,
            )

    def stop(self, timeout: float = 3.0) -> None:
        """Pide la detención del hilo y espera a que termine."""
        self._stop_event.set()
        if self.is_alive():
            self.join(timeout=timeout)

    # -- Estado ------------------------------------------------------------
    @property
    def fps(self) -> float:
        """FPS de procesamiento."""
        return self._fps_meter.fps

    @property
    def process_ms(self) -> float:
        """Tiempo promedio de MediaPipe por frame."""
        return self._process_meter.milliseconds

    @property
    def inference_ms(self) -> float:
        """Tiempo promedio del modelo por ventana."""
        return self._inference_meter.milliseconds

    @property
    def error(self) -> str | None:
        """Mensaje de error en español si el hilo murió."""
        return self._error
