"""Grabador de dataset de LSP (Fase 1).

Abre la ventana de grabación: vista de cámara con landmarks, selector de seña,
cuenta regresiva, control de calidad automático y guardado en
``dataset/raw/<seña>/pXX_sYY_rZZ.npz``.

Uso::

    python scripts/record_dataset.py
    python scripts/record_dataset.py --persona p02
    python scripts/record_dataset.py --camara 1
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ / "src") not in sys.path:
    sys.path.insert(0, str(RAIZ / "src"))

from senasperu.config import Config, ConfigError, load_config  # noqa: E402
from senasperu.logging_setup import setup_logging  # noqa: E402

PERSONA_VALIDA = re.compile(r"^p\d{2}$")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define y procesa los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="Grabador de dataset de LSP (Fase 1).")
    parser.add_argument("--config", type=Path, default=None, help="Ruta a un YAML alternativo.")
    parser.add_argument(
        "--persona",
        type=str,
        default=None,
        help="Identificador de la persona señante (pXX). Por defecto, el del YAML.",
    )
    parser.add_argument(
        "--camara", type=int, default=None, help="Índice de cámara (sobrescribe el YAML)."
    )
    parser.add_argument(
        "--video", type=Path, default=None, help="Archivo de video en lugar de la webcam."
    )
    return parser.parse_args(argv)


def apply_overrides(config: Config, args: argparse.Namespace) -> Config:
    """Devuelve una configuración con los ajustes de línea de comandos."""
    datos = config.to_dict()
    if args.camara is not None:
        datos["camara"]["indice"] = args.camara
    return Config(datos)


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada. Devuelve el código de salida del proceso."""
    args = parse_args(argv)

    if args.persona is not None and not PERSONA_VALIDA.match(args.persona):
        print(
            f"El identificador de persona '{args.persona}' debe tener el formato pXX "
            "(por ejemplo p01).",
            file=sys.stderr,
        )
        return 2

    try:
        config = apply_overrides(load_config(args.config), args)
    except ConfigError as error:
        print(f"Error de configuración: {error}", file=sys.stderr)
        return 2

    logger = setup_logging(config)
    logger.info("Iniciando grabador de dataset")

    from PySide6.QtWidgets import QApplication

    from senasperu.ui.recorder_window import RecorderWindow

    app = QApplication(sys.argv[:1])
    window = RecorderWindow(config, person=args.persona, video_path=args.video)
    window.show()
    if not window.start():
        return 1
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
