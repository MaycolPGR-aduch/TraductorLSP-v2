"""Evaluación del modelo con splits por sesión (Fase 2).

**Los splits son SIEMPRE por sesión completa, nunca por repetición.** Dos
repeticiones grabadas en la misma sesión comparten iluminación, ropa, encuadre y
el estado del señante ese día. Repartirlas entre entrenamiento y prueba deja que
el modelo reconozca la sesión en lugar de la seña: la precisión sube, parece
excelente, y se desploma con un usuario real. Por eso ``split_by_session`` agrupa
por ``(persona, sesión)`` y exige que existan al menos dos sesiones.

Produce precisión global, precisión por seña y matriz de confusión.

Uso::

    python scripts/evaluate.py
    python scripts/evaluate.py --checkpoint models/senasperu.pt --matriz confusion.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ / "src") not in sys.path:
    sys.path.insert(0, str(RAIZ / "src"))

import torch  # noqa: E402

from senasperu.config import Config, ConfigError, load_config  # noqa: E402
from senasperu.data.dataset import (  # noqa: E402
    build_window_set,
    scan_recordings,
    split_by_session,
)
from senasperu.features.vector import layout_from_config  # noqa: E402
from senasperu.logging_setup import setup_logging  # noqa: E402
from senasperu.model.architecture import build_model  # noqa: E402
from senasperu.vocabulary import load_vocabulary  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define y procesa los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="Evalúa el modelo con split por sesión.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument(
        "--matriz", type=Path, default=None, help="Ruta donde guardar la matriz de confusión."
    )
    return parser.parse_args(argv)


def confusion_matrix(reales: np.ndarray, predichas: np.ndarray, clases: int) -> np.ndarray:
    """Matriz de confusión ``(real, predicha)``."""
    matriz = np.zeros((clases, clases), dtype=np.int32)
    for real, predicha in zip(reales, predichas):
        matriz[int(real), int(predicha)] += 1
    return matriz


def print_report(matriz: np.ndarray, nombres: list[str]) -> float:
    """Imprime precisión por seña y devuelve la precisión global."""
    total = int(matriz.sum())
    aciertos = int(np.trace(matriz))
    print(f"\nPrecisión global: {aciertos / max(1, total):.3f} ({aciertos}/{total})\n")
    print(f"{'seña':<16}{'muestras':>9}{'aciertos':>9}{'precisión':>11}   principal confusión")
    for indice, nombre in enumerate(nombres):
        muestras = int(matriz[indice].sum())
        if muestras == 0:
            print(f"{nombre:<16}{0:>9}{'-':>9}{'sin datos':>11}")
            continue
        correctos = int(matriz[indice, indice])
        fila = matriz[indice].copy()
        fila[indice] = 0
        peor = int(np.argmax(fila))
        detalle = f"{nombres[peor]} ({int(fila[peor])})" if fila[peor] > 0 else "-"
        print(
            f"{nombre:<16}{muestras:>9}{correctos:>9}{correctos / muestras:>11.3f}   {detalle}"
        )
    return aciertos / max(1, total)


def save_matrix_plot(matriz: np.ndarray, nombres: list[str], ruta: Path) -> None:
    """Guarda la matriz de confusión como imagen."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    normalizada = matriz / np.maximum(1, matriz.sum(axis=1, keepdims=True))
    figura, ejes = plt.subplots(figsize=(max(6, len(nombres) * 0.5),) * 2)
    imagen = ejes.imshow(normalizada, cmap="Blues", vmin=0.0, vmax=1.0)
    ejes.set_xticks(range(len(nombres)), nombres, rotation=90, fontsize=7)
    ejes.set_yticks(range(len(nombres)), nombres, fontsize=7)
    ejes.set_xlabel("Predicha")
    ejes.set_ylabel("Real")
    ejes.set_title("Matriz de confusión (split por sesión)")
    figura.colorbar(imagen, ax=ejes, fraction=0.046)
    figura.tight_layout()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(ruta, dpi=150)
    plt.close(figura)


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada de la evaluación."""
    args = parse_args(argv)
    try:
        config = load_config(args.config)
    except ConfigError as error:
        print(f"Error de configuración: {error}", file=sys.stderr)
        return 2
    if args.dataset is not None:
        datos = config.to_dict()
        datos["dataset"]["ruta_raiz"] = str(args.dataset)
        config = Config(datos)

    logger = setup_logging(config)
    ruta_checkpoint = Path(args.checkpoint) if args.checkpoint else RAIZ / "models" / "senasperu.pt"
    if not ruta_checkpoint.is_file():
        logger.error("No se encontró el checkpoint %s.", ruta_checkpoint)
        return 1

    grabaciones = scan_recordings(config.resolve_path("dataset.ruta_raiz"))
    if not grabaciones:
        logger.error("No hay grabaciones que evaluar.")
        return 1

    semilla = int(config.get("dataset.semilla", 42))
    try:
        _, prueba = split_by_session(
            grabaciones,
            test_ratio=float(config.require("dataset.proporcion_test")),
            seed=semilla,
        )
    except ValueError as error:
        logger.error("%s", error)
        return 1

    vocabulario = load_vocabulary(config)
    ventanas = build_window_set(
        prueba, config, vocabulary=vocabulario, layout=layout_from_config(config)
    )
    if not len(ventanas):
        logger.error("El conjunto de prueba no produjo ventanas utilizables.")
        return 1

    checkpoint = torch.load(ruta_checkpoint, map_location="cpu", weights_only=False)
    datos = config.to_dict()
    datos["modelo"]["arquitectura"] = checkpoint.get("architecture", "transformer")
    modelo = build_model(Config(datos), checkpoint["input_size"], checkpoint["num_classes"])
    modelo.load_state_dict(checkpoint["state_dict"])
    modelo.eval()

    with torch.no_grad():
        logits = modelo(torch.from_numpy(ventanas.features))
    predichas = logits.argmax(dim=1).numpy()

    nombres = [sign.glosa for sign in vocabulario]
    matriz = confusion_matrix(ventanas.labels, predichas, len(vocabulario))
    precision = print_report(matriz, nombres)

    sesiones = sorted({clave for clave in ventanas.session_keys})
    print(f"\nSesiones de prueba: {sesiones}")
    print(f"Ventanas evaluadas: {len(ventanas)}")

    if args.matriz is not None:
        save_matrix_plot(matriz, nombres, args.matriz)
        print(f"Matriz de confusión guardada en: {args.matriz}")

    objetivo = 0.9
    print(f"\nCriterio de Fase 2: precisión > {objetivo:.0%} -> ", end="")
    print("OK" if precision > objetivo else "NO CUMPLE")
    return 0 if precision > objetivo else 1


if __name__ == "__main__":
    raise SystemExit(main())
