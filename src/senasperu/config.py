"""Carga y acceso a la configuración central del proyecto.

Todos los parámetros del sistema viven en ``config/default.yaml``. Este módulo es
la única puerta de entrada a esos valores: ningún otro módulo debe contener
números mágicos ni leer el YAML por su cuenta.

Uso típico::

    from senasperu.config import load_config

    config = load_config()
    ancho = config.camara.ancho            # acceso por atributo
    fps = config.get("camara.fps_objetivo")  # acceso por ruta punteada
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import yaml

# Raíz del repositorio: src/senasperu/config.py -> senasperu -> src -> raíz
ROOT_PATH: Path = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH: Path = ROOT_PATH / "config" / "default.yaml"
CONFIG_ENV_VAR: str = "SENASPERU_CONFIG"

# Secciones que deben existir sí o sí; si falta alguna, fallamos al arrancar
# y no a mitad de una grabación.
REQUIRED_SECTIONS: tuple[str, ...] = (
    "proyecto",
    "camara",
    "mediapipe",
    "normalizacion",
    "ventana",
    "grabador",
    "calidad_datos",
    "dataset",
    "vocabulario",
    "modelo",
    "entrenamiento",
    "inferencia",
    "estabilizacion",
    "tts",
    "ui",
    "logging",
)


class ConfigError(RuntimeError):
    """Error de carga o validación de la configuración."""


class Config(Mapping[str, Any]):
    """Vista de solo lectura sobre el árbol de configuración.

    Permite acceso por atributo (``config.camara.ancho``), por clave
    (``config["camara"]["ancho"]``) y por ruta punteada
    (``config.get("camara.ancho")``). Los sub-diccionarios se envuelven
    automáticamente en otro :class:`Config`.
    """

    __slots__ = ("_data", "_prefix")

    def __init__(self, data: Mapping[str, Any], prefix: str = "") -> None:
        self._data: dict[str, Any] = dict(data)
        self._prefix: str = prefix

    # -- Protocolo Mapping -------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        value = self._data[key]
        return self._wrap(key, value)

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    # -- Acceso cómodo -----------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            ruta = f"{self._prefix}{name}"
            raise AttributeError(
                f"La clave de configuración '{ruta}' no existe en el archivo YAML."
            ) from None

    def get(self, path: str, default: Any = None) -> Any:
        """Devuelve el valor en la ruta punteada indicada, o ``default`` si no existe.

        Args:
            path: Ruta tipo ``"camara.fps_objetivo"``.
            default: Valor a devolver si la ruta no existe.
        """
        actual: Any = self
        for parte in path.split("."):
            if isinstance(actual, Config) and parte in actual._data:
                actual = actual[parte]
            elif isinstance(actual, Mapping) and parte in actual:
                actual = actual[parte]
            else:
                return default
        return actual

    def require(self, path: str) -> Any:
        """Igual que :meth:`get`, pero lanza :class:`ConfigError` si la ruta falta."""
        centinela = object()
        valor = self.get(path, centinela)
        if valor is centinela:
            raise ConfigError(f"Falta el parámetro obligatorio '{path}' en la configuración.")
        return valor

    def resolve_path(self, path: str) -> Path:
        """Convierte una ruta relativa de la configuración en ruta absoluta del repo.

        Args:
            path: Ruta punteada cuyo valor es una ruta de archivo o carpeta
                (por ejemplo ``"dataset.ruta_raiz"``).
        """
        valor = Path(str(self.require(path)))
        return valor if valor.is_absolute() else (ROOT_PATH / valor)

    def to_dict(self) -> dict[str, Any]:
        """Copia mutable del árbol de configuración (útil para tests y logs)."""
        return _deep_copy(self._data)

    def _wrap(self, key: str, value: Any) -> Any:
        if isinstance(value, Mapping):
            return Config(value, prefix=f"{self._prefix}{key}.")
        if isinstance(value, list):
            return [
                Config(item, prefix=f"{self._prefix}{key}[].") if isinstance(item, Mapping) else item
                for item in value
            ]
        return value

    def __repr__(self) -> str:  # pragma: no cover - solo diagnóstico
        return f"Config({sorted(self._data)})"


def _deep_copy(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _deep_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_deep_copy(v) for v in value]
    return value


def load_config(path: str | os.PathLike[str] | None = None, *, validate: bool = True) -> Config:
    """Carga la configuración del proyecto desde YAML.

    El orden de resolución es: ``path`` explícito, variable de entorno
    ``SENASPERU_CONFIG`` y, por último, ``config/default.yaml``.

    Args:
        path: Ruta al archivo YAML. Si es ``None``, se resuelve automáticamente.
        validate: Si es ``True``, verifica que existan todas las secciones obligatorias.

    Raises:
        ConfigError: Si el archivo no existe, no es un YAML válido o faltan secciones.
    """
    ruta = Path(path) if path is not None else _default_config_path()
    if not ruta.is_file():
        raise ConfigError(f"No se encontró el archivo de configuración: {ruta}")

    try:
        with ruta.open("r", encoding="utf-8") as archivo:
            datos = yaml.safe_load(archivo)
    except yaml.YAMLError as error:
        raise ConfigError(f"El archivo de configuración '{ruta}' no es un YAML válido: {error}") from error

    if not isinstance(datos, Mapping):
        raise ConfigError(f"El archivo de configuración '{ruta}' debe contener un diccionario.")

    config = Config(datos)
    if validate:
        validate_config(config)
    return config


def validate_config(config: Config) -> None:
    """Verifica que la configuración tenga las secciones y claves críticas.

    Raises:
        ConfigError: Si falta alguna sección obligatoria o el vocabulario es inválido.
    """
    faltantes = [seccion for seccion in REQUIRED_SECTIONS if seccion not in config]
    if faltantes:
        raise ConfigError(
            "Faltan secciones obligatorias en la configuración: " + ", ".join(faltantes)
        )

    vocabulario = config["vocabulario"]
    if not isinstance(vocabulario, list) or not vocabulario:
        raise ConfigError("La sección 'vocabulario' debe ser una lista no vacía.")

    ids: list[str] = []
    for entrada in vocabulario:
        if not isinstance(entrada, Mapping):
            raise ConfigError("Cada entrada de 'vocabulario' debe ser un diccionario.")
        for clave in ("id", "glosa", "texto", "espejable"):
            if clave not in entrada:
                raise ConfigError(
                    f"La entrada de vocabulario {dict(entrada)!r} no tiene la clave '{clave}'."
                )
        ids.append(str(entrada["id"]))

    duplicados = sorted({i for i in ids if ids.count(i) > 1})
    if duplicados:
        raise ConfigError("Hay ids de seña duplicados en 'vocabulario': " + ", ".join(duplicados))

    # La clase de reposo es obligatoria: sin ella la app produce traducciones espurias.
    if "no_sena" not in ids:
        raise ConfigError(
            "El vocabulario debe incluir la clase de reposo con id 'no_sena'."
        )


def _default_config_path() -> Path:
    desde_entorno = os.environ.get(CONFIG_ENV_VAR)
    if desde_entorno:
        return Path(desde_entorno)
    return DEFAULT_CONFIG_PATH
