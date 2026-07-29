"""Capa de estabilización sobre las predicciones crudas del modelo.

Sin esto la traducción parpadea: el modelo emite una predicción cada pocos
frames y basta un fotograma raro para que aparezca una palabra que nadie señó.
Se aplican cuatro filtros, en este orden:

1. **Umbral de confianza.** Lo que no llega al umbral no vota.
2. **Votación por mayoría** sobre las últimas N ventanas.
3. **Debouncing.** Una seña se confirma solo si domina durante un tiempo mínimo.
4. **Paso por reposo.** Una seña confirmada no se repite hasta que el señante
   vuelve a la posición de descanso. Sin esta regla, sostener una seña dos
   segundos la escribiría cinco veces.

Es lógica pura: sin modelo, sin cámara y sin Qt, así que se testea entera.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

from senasperu.config import Config


@dataclass(frozen=True, slots=True)
class StabilizerState:
    """Estado visible del estabilizador, para mostrar en la interfaz.

    Attributes:
        confirmed: Índice de la seña recién confirmada, o ``None``. Solo se
            entrega **una vez** por confirmación.
        candidate: Índice de la seña que va ganando la votación, o ``None``.
        candidate_confidence: Confianza media del candidato.
        progress: Avance del debounce (0.0-1.0); sirve para una barra en la UI.
        at_rest: ``True`` si el señante está en reposo.
    """

    confirmed: int | None = None
    candidate: int | None = None
    candidate_confidence: float = 0.0
    progress: float = 0.0
    at_rest: bool = False


class Stabilizer:
    """Convierte predicciones crudas en confirmaciones estables."""

    def __init__(
        self,
        *,
        rest_index: int,
        confidence_threshold: float,
        vote_windows: int,
        debounce_seconds: float,
        require_rest_between_repeats: bool = True,
    ) -> None:
        """Args:
        rest_index: Índice de la clase de reposo en el vocabulario.
        confidence_threshold: Confianza mínima para que una predicción vote.
        vote_windows: Cuántas ventanas recientes participan en la votación.
        debounce_seconds: Tiempo que una seña debe dominar para confirmarse.
        require_rest_between_repeats: Si una seña confirmada necesita pasar por
            reposo antes de poder repetirse.
        """
        self._rest_index = rest_index
        self._threshold = float(confidence_threshold)
        self._votes: deque[int | None] = deque(maxlen=max(1, vote_windows))
        self._debounce = float(debounce_seconds)
        self._require_rest = require_rest_between_repeats

        self._candidate: int | None = None
        self._candidate_since: float = 0.0
        self._candidate_confidences: list[float] = []
        self._last_confirmed: int | None = None
        self._seen_rest_since_confirm = True

    @classmethod
    def from_config(cls, config: Config, rest_index: int) -> Stabilizer:
        """Construye el estabilizador con la sección ``estabilizacion`` del YAML."""
        return cls(
            rest_index=rest_index,
            confidence_threshold=float(config.require("estabilizacion.umbral_confianza")),
            vote_windows=int(config.require("estabilizacion.ventanas_votacion")),
            debounce_seconds=float(config.require("estabilizacion.debounce_segundos")),
            require_rest_between_repeats=bool(
                config.get("estabilizacion.requiere_reposo_para_repetir", True)
            ),
        )

    def update(self, class_index: int, confidence: float, timestamp: float) -> StabilizerState:
        """Procesa una predicción cruda y devuelve el estado resultante.

        Args:
            class_index: Clase predicha por el modelo.
            confidence: Confianza de esa predicción (0.0-1.0).
            timestamp: Marca de tiempo monótona de la predicción, en segundos.

        Returns:
            El estado; ``confirmed`` trae la seña solo en el instante en que se
            confirma, nunca repetida.
        """
        # 1. Umbral: lo que no convence, no vota.
        voto = class_index if confidence >= self._threshold else None
        self._votes.append(voto)

        # 2. Votación por mayoría sobre las últimas ventanas.
        ganador = self._majority()

        if ganador == self._rest_index:
            self._seen_rest_since_confirm = True

        # 3. Debouncing: el candidato debe mantenerse en el tiempo.
        if ganador != self._candidate:
            self._candidate = ganador
            self._candidate_since = timestamp
            self._candidate_confidences = []
        if ganador is not None and confidence >= self._threshold and class_index == ganador:
            self._candidate_confidences.append(confidence)

        estado_base = StabilizerState(
            candidate=ganador,
            candidate_confidence=self._mean_confidence(),
            progress=self._progress(timestamp),
            at_rest=ganador == self._rest_index,
        )
        if ganador is None or ganador == self._rest_index:
            return estado_base
        if timestamp - self._candidate_since < self._debounce:
            return estado_base

        # 4. Una seña confirmada no se repite sin pasar por reposo.
        if self._require_rest and ganador == self._last_confirmed:
            if not self._seen_rest_since_confirm:
                return estado_base

        self._last_confirmed = ganador
        self._seen_rest_since_confirm = False
        # Se reinicia el reloj para que la misma seña sostenida no se confirme
        # de nuevo en el frame siguiente.
        self._candidate_since = timestamp
        return StabilizerState(
            confirmed=ganador,
            candidate=ganador,
            candidate_confidence=self._mean_confidence(),
            progress=1.0,
            at_rest=False,
        )

    def reset(self) -> None:
        """Olvida el historial. Se usa al limpiar la conversación."""
        self._votes.clear()
        self._candidate = None
        self._candidate_since = 0.0
        self._candidate_confidences = []
        self._last_confirmed = None
        self._seen_rest_since_confirm = True

    def _majority(self) -> int | None:
        """Clase más votada, o ``None`` si no hay mayoría clara."""
        validos = [voto for voto in self._votes if voto is not None]
        if not validos:
            return None
        conteo = Counter(validos)
        clase, cantidad = conteo.most_common(1)[0]
        # Mayoría sobre el total de ventanas, no solo sobre las que votaron: si
        # la mitad de las ventanas no llegaron al umbral, no hay nada seguro.
        if cantidad * 2 <= len(self._votes):
            return None
        return int(clase)

    def _progress(self, timestamp: float) -> float:
        if self._candidate is None or self._candidate == self._rest_index:
            return 0.0
        if self._debounce <= 0:
            return 1.0
        return min(1.0, (timestamp - self._candidate_since) / self._debounce)

    def _mean_confidence(self) -> float:
        if not self._candidate_confidences:
            return 0.0
        return sum(self._candidate_confidences) / len(self._candidate_confidences)
