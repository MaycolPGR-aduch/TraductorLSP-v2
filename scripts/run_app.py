"""App de traducción de LSP en tiempo real (Fase 3).

Abre la cámara, reconoce señas y va escribiendo la traducción en pantalla, con
opción de reproducirla con voz.

Uso::

    python scripts/run_app.py
    python scripts/run_app.py --modelo models/senasperu_int8.onnx
    python scripts/run_app.py --video ruta/al/video.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ / "src") not in sys.path:
    sys.path.insert(0, str(RAIZ / "src"))

from senasperu.config import Config, ConfigError, load_config  # noqa: E402
from senasperu.logging_setup import setup_logging  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define y procesa los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="Traductor de LSP en tiempo real.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--camara", type=int, default=None, help="Índice de cámara.")
    parser.add_argument("--modelo", type=Path, default=None, help="Ruta del modelo ONNX.")
    parser.add_argument("--video", type=Path, default=None, help="Video en lugar de la webcam.")
    return parser.parse_args(argv)


def apply_overrides(config: Config, args: argparse.Namespace) -> Config:
    """Aplica los ajustes de línea de comandos."""
    datos = config.to_dict()
    if args.camara is not None:
        datos["camara"]["indice"] = args.camara
    if args.modelo is not None:
        datos["inferencia"]["ruta_modelo_onnx"] = str(args.modelo)
    return Config(datos)


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada de la aplicación."""
    args = parse_args(argv)
    try:
        config = apply_overrides(load_config(args.config), args)
    except ConfigError as error:
        print(f"Error de configuración: {error}", file=sys.stderr)
        return 2

    logger = setup_logging(config)

    modelo = config.resolve_path("inferencia.ruta_modelo_onnx")
    if not modelo.is_file():
        mensaje = (
            f"No se encontró el modelo en {modelo}.\n"
            "Antes de traducir hay que grabar el dataset y entrenar:\n"
            "  1. python scripts/record_dataset.py\n"
            "  2. python scripts/train.py\n"
            "  3. python scripts/export_onnx.py"
        )
        logger.error("%s", mensaje)
        print(mensaje, file=sys.stderr)
        return 1

    logger.info("Iniciando la app de traducción")

    from PySide6.QtWidgets import QApplication

    from senasperu.ui.translator_window import TranslatorWindow

    app = QApplication(sys.argv[:1])
    window = TranslatorWindow(config, video_path=args.video)
    window.show()
    if not window.start():
        return 1
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
