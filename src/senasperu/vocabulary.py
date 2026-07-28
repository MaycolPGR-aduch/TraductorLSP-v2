"""Vocabulario de señas, leído de la configuración.

La lista de señas vive en ``config/default.yaml`` y es la única fuente de
verdad: el grabador, el entrenamiento y la app de traducción trabajan sobre
estos mismos ids y en este mismo orden (el orden define los índices de clase
del modelo, así que **no se deben reordenar** las entradas del YAML una vez
grabado el dataset).
"""

from __future__ import annotations

from dataclasses import dataclass

from senasperu.config import Config

# Id de la clase de reposo. Es obligatoria: sin ella el modelo traduce cualquier
# movimiento cotidiano.
REST_SIGN_ID: str = "no_sena"


@dataclass(frozen=True, slots=True)
class Sign:
    """Una seña del vocabulario.

    Attributes:
        id: Identificador estable, usado en nombres de archivo y etiquetas.
        glosa: Nombre en mayúsculas, como se escribe en glosas de LSP.
        text: Texto en español que produce la traducción.
        mirrorable: Si la seña admite aumento de datos por espejado horizontal.
        index: Posición en el vocabulario (índice de clase del modelo).
    """

    id: str
    glosa: str
    text: str
    mirrorable: bool
    index: int

    @property
    def is_rest(self) -> bool:
        """``True`` si es la clase de reposo/no-seña."""
        return self.id == REST_SIGN_ID


def load_vocabulary(config: Config) -> tuple[Sign, ...]:
    """Lee el vocabulario de la configuración.

    Returns:
        Las señas en el orden declarado en el YAML.
    """
    signs: list[Sign] = []
    for indice, entrada in enumerate(config.vocabulario):
        signs.append(
            Sign(
                id=str(entrada["id"]),
                glosa=str(entrada["glosa"]),
                text=str(entrada["texto"]),
                mirrorable=bool(entrada["espejable"]),
                index=indice,
            )
        )
    return tuple(signs)


def find_sign(vocabulary: tuple[Sign, ...], sign_id: str) -> Sign | None:
    """Busca una seña por su id; ``None`` si no está en el vocabulario."""
    for sign in vocabulary:
        if sign.id == sign_id:
            return sign
    return None
