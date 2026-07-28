"""Smoke test de Fase 0: cámara + MediaPipe Holistic + PySide6.

Abre una ventana con la vista de la cámara, el esqueleto de landmarks superpuesto
y las métricas de rendimiento necesarias para validar los criterios de la fase.

Uso::

    python scripts/smoke_test.py
    python scripts/smoke_test.py --camara 1
    python scripts/smoke_test.py --video ruta/al/video.mp4   # prueba sin webcam
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ / "src") not in sys.path:
    # Permite ejecutar el script sin haber hecho 'pip install -e .'
    sys.path.insert(0, str(RAIZ / "src"))

from senasperu.config import Config, ConfigError, load_config  # noqa: E402
from senasperu.logging_setup import setup_logging  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define y procesa los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Prueba de cámara, MediaPipe y ventana Qt (Fase 0)."
    )
    parser.add_argument("--config", type=Path, default=None, help="Ruta a un YAML alternativo.")
    parser.add_argument(
        "--camara", type=int, default=None, help="Índice de cámara (sobrescribe el YAML)."
    )
    parser.add_argument(
        "--video", type=Path, default=None, help="Archivo de video en lugar de la webcam."
    )
    parser.add_argument(
        "--sin-landmarks",
        action="store_true",
        help="Arranca sin dibujar el esqueleto (para medir el costo del dibujo).",
    )
    return parser.parse_args(argv)


def apply_overrides(config: Config, args: argparse.Namespace) -> Config:
    """Devuelve una configuración con los ajustes pedidos por línea de comandos."""
    datos = config.to_dict()
    if args.camara is not None:
        datos["camara"]["indice"] = args.camara
    if args.sin_landmarks:
        datos["ui"]["mostrar_landmarks"] = False
    return Config(datos)


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada. Devuelve el código de salida del proceso."""
    args = parse_args(argv)

    try:
        config = apply_overrides(load_config(args.config), args)
    except ConfigError as error:
        print(f"Error de configuración: {error}", file=sys.stderr)
        return 2

    logger = setup_logging(config)
    logger.info("Iniciando smoke test de Fase 0")

    from PySide6.QtWidgets import QApplication

    from senasperu.ui.smoke_window import SmokeWindow

    app = QApplication(sys.argv[:1])
    window = SmokeWindow(config, video_path=args.video)
    window.show()
    if not window.start():
        return 1
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
