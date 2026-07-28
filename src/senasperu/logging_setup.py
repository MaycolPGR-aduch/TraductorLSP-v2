"""Configuración de logging del proyecto.

Escribe a consola y a un archivo rotativo. Los mensajes dirigidos al usuario van
en español; el nivel y la ruta salen de la sección ``logging`` del YAML.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from senasperu.config import Config

LOG_FORMAT: str = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
BACKUP_COUNT: int = 3


def setup_logging(config: Config, *, force: bool = False) -> logging.Logger:
    """Configura el logger raíz según la configuración y devuelve el logger de la app.

    Args:
        config: Configuración cargada con :func:`senasperu.config.load_config`.
        force: Si es ``True``, reemplaza los handlers existentes (útil en tests).

    Returns:
        El logger ``senasperu``, listo para usarse.
    """
    root = logging.getLogger()
    if root.handlers and not force:
        return logging.getLogger("senasperu")
    if force:
        for handler in list(root.handlers):
            root.removeHandler(handler)
            handler.close()

    nivel_texto = str(config.get("logging.nivel", "INFO")).upper()
    nivel = getattr(logging, nivel_texto, logging.INFO)
    root.setLevel(nivel)

    formato = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    consola = logging.StreamHandler(stream=sys.stderr)
    consola.setFormatter(formato)
    root.addHandler(consola)

    ruta_log = _log_path(config)
    if ruta_log is not None:
        rotacion_mb = float(config.get("logging.rotacion_mb", 5))
        archivo = RotatingFileHandler(
            ruta_log,
            maxBytes=int(rotacion_mb * 1024 * 1024),
            backupCount=BACKUP_COUNT,
            encoding="utf-8",
        )
        archivo.setFormatter(formato)
        root.addHandler(archivo)

    logger = logging.getLogger("senasperu")
    logger.debug("Logging configurado en nivel %s (archivo: %s)", nivel_texto, ruta_log)
    return logger


def _log_path(config: Config) -> Path | None:
    """Devuelve la ruta del archivo de log, creando su carpeta; ``None`` si no se puede."""
    valor = config.get("logging.archivo")
    if not valor:
        return None
    try:
        ruta = config.resolve_path("logging.archivo")
        ruta.parent.mkdir(parents=True, exist_ok=True)
        return ruta
    except OSError as error:
        logging.getLogger("senasperu").warning(
            "No se pudo crear el archivo de log '%s': %s. Se registrará solo en consola.",
            valor,
            error,
        )
        return None
