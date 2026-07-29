"""Entrenamiento del clasificador de señas (Fase 2).

Reproducible: semillas fijas y split por sesión completa. Pensado para correr
igual en local que en Google Colab.

Uso::

    python scripts/train.py
    python scripts/train.py --epocas 50 --arquitectura bilstm
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ / "src") not in sys.path:
    sys.path.insert(0, str(RAIZ / "src"))

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from senasperu.config import Config, ConfigError, load_config  # noqa: E402
from senasperu.data.augment import WindowAugmenter  # noqa: E402
from senasperu.data.dataset import (  # noqa: E402
    WindowSet,
    build_window_set,
    scan_recordings,
    split_by_session,
)
from senasperu.features.vector import layout_from_config  # noqa: E402
from senasperu.logging_setup import setup_logging  # noqa: E402
from senasperu.model.architecture import build_model, count_parameters  # noqa: E402
from senasperu.vocabulary import load_vocabulary  # noqa: E402


class WindowDataset(Dataset):
    """Ventanas con aumento aplicado al vuelo (solo en entrenamiento)."""

    def __init__(self, window_set: WindowSet, augmenter: WindowAugmenter | None) -> None:
        self._windows = window_set
        self._augmenter = augmenter

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, indice: int):  # noqa: D105
        features = self._windows.features[indice]
        if self._augmenter is not None:
            features = self._augmenter.augment(
                features, mirrorable=bool(self._windows.mirrorable[indice])
            )
        return torch.from_numpy(np.ascontiguousarray(features)), int(
            self._windows.labels[indice]
        )


def set_seed(seed: int) -> None:
    """Fija todas las semillas para que el entrenamiento sea reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define y procesa los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="Entrena el clasificador de señas.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, default=None, help="Raíz alternativa del dataset.")
    parser.add_argument("--salida", type=Path, default=None, help="Carpeta de checkpoints.")
    parser.add_argument("--epocas", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument(
        "--arquitectura", type=str, default=None, choices=["transformer", "bilstm"]
    )
    return parser.parse_args(argv)


def apply_overrides(config: Config, args: argparse.Namespace) -> Config:
    """Aplica los ajustes de línea de comandos sobre la configuración."""
    datos = config.to_dict()
    if args.epocas is not None:
        datos["entrenamiento"]["epocas"] = args.epocas
    if args.batch is not None:
        datos["entrenamiento"]["batch_size"] = args.batch
    if args.arquitectura is not None:
        datos["modelo"]["arquitectura"] = args.arquitectura
    if args.dataset is not None:
        datos["dataset"]["ruta_raiz"] = str(args.dataset)
    return Config(datos)


def evaluate(model: nn.Module, loader: DataLoader, criterio: nn.Module) -> tuple[float, float]:
    """Devuelve ``(pérdida, precisión)`` sobre un conjunto."""
    model.eval()
    perdida_total, aciertos, total = 0.0, 0, 0
    with torch.no_grad():
        for lote, etiquetas in loader:
            logits = model(lote)
            perdida_total += float(criterio(logits, etiquetas)) * lote.shape[0]
            aciertos += int((logits.argmax(dim=1) == etiquetas).sum())
            total += lote.shape[0]
    if total == 0:
        return 0.0, 0.0
    return perdida_total / total, aciertos / total


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada del entrenamiento."""
    args = parse_args(argv)
    try:
        config = apply_overrides(load_config(args.config), args)
    except ConfigError as error:
        print(f"Error de configuración: {error}", file=sys.stderr)
        return 2

    logger = setup_logging(config)
    semilla = int(config.get("entrenamiento.semilla", 42))
    set_seed(semilla)

    raiz = config.resolve_path("dataset.ruta_raiz")
    grabaciones = scan_recordings(raiz)
    if not grabaciones:
        logger.error("No hay grabaciones en %s. Graba el dataset primero.", raiz)
        return 1
    logger.info("Grabaciones encontradas: %s", len(grabaciones))

    try:
        entrenamiento, prueba = split_by_session(
            grabaciones,
            test_ratio=float(config.require("dataset.proporcion_test")),
            seed=semilla,
        )
    except ValueError as error:
        logger.error("%s", error)
        return 1

    vocabulario = load_vocabulary(config)
    layout = layout_from_config(config)
    ventanas_train = build_window_set(entrenamiento, config, vocabulary=vocabulario, layout=layout)
    ventanas_test = build_window_set(prueba, config, vocabulary=vocabulario, layout=layout)
    if not len(ventanas_train) or not len(ventanas_test):
        logger.error("El split no produjo ventanas utilizables en train o en test.")
        return 1

    logger.info(
        "Ventanas: %s de entrenamiento, %s de prueba, %s features por frame",
        len(ventanas_train),
        len(ventanas_test),
        ventanas_train.features.shape[2],
    )

    augmentador = WindowAugmenter.from_config(config, layout)
    batch = int(config.require("entrenamiento.batch_size"))
    cargador_train = DataLoader(
        WindowDataset(ventanas_train, augmentador), batch_size=batch, shuffle=True
    )
    cargador_test = DataLoader(WindowDataset(ventanas_test, None), batch_size=batch)

    modelo = build_model(config, ventanas_train.features.shape[2], len(vocabulario))
    logger.info(
        "Modelo %s con %s parámetros",
        config.get("modelo.arquitectura"),
        f"{count_parameters(modelo):,}",
    )

    criterio = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizador = torch.optim.AdamW(
        modelo.parameters(),
        lr=float(config.require("entrenamiento.learning_rate")),
        weight_decay=float(config.require("entrenamiento.weight_decay")),
    )
    epocas = int(config.require("entrenamiento.epocas"))
    paciencia = int(config.require("entrenamiento.early_stopping_paciencia"))
    planificador = torch.optim.lr_scheduler.CosineAnnealingLR(optimizador, T_max=max(1, epocas))

    salida = Path(args.salida) if args.salida else (RAIZ / "models")
    salida.mkdir(parents=True, exist_ok=True)
    ruta_checkpoint = salida / "senasperu.pt"

    mejor_precision, mejor_epoca, sin_mejora = 0.0, 0, 0
    inicio = time.perf_counter()
    for epoca in range(1, epocas + 1):
        modelo.train()
        perdida_total, total = 0.0, 0
        for lote, etiquetas in cargador_train:
            optimizador.zero_grad()
            perdida = criterio(modelo(lote), etiquetas)
            perdida.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            optimizador.step()
            perdida_total += float(perdida) * lote.shape[0]
            total += lote.shape[0]
        planificador.step()

        perdida_test, precision_test = evaluate(modelo, cargador_test, criterio)
        logger.info(
            "Época %3d/%d | pérdida train %.4f | pérdida test %.4f | precisión test %.3f",
            epoca,
            epocas,
            perdida_total / max(1, total),
            perdida_test,
            precision_test,
        )

        if precision_test > mejor_precision:
            mejor_precision, mejor_epoca, sin_mejora = precision_test, epoca, 0
            torch.save(
                {
                    "state_dict": modelo.state_dict(),
                    "input_size": ventanas_train.features.shape[2],
                    "frames_per_window": ventanas_train.features.shape[1],
                    "num_classes": len(vocabulario),
                    "architecture": config.get("modelo.arquitectura"),
                    "labels": [sign.id for sign in vocabulario],
                    "accuracy": precision_test,
                    "epoch": epoca,
                },
                ruta_checkpoint,
            )
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia:
                logger.info("Early stopping en la época %s.", epoca)
                break

    duracion = time.perf_counter() - inicio
    logger.info(
        "Entrenamiento terminado en %.1f s. Mejor precisión %.3f (época %s). Checkpoint: %s",
        duracion,
        mejor_precision,
        mejor_epoca,
        ruta_checkpoint,
    )
    (salida / "entrenamiento.json").write_text(
        json.dumps(
            {
                "mejor_precision": mejor_precision,
                "mejor_epoca": mejor_epoca,
                "ventanas_entrenamiento": len(ventanas_train),
                "ventanas_prueba": len(ventanas_test),
                "duracion_segundos": round(duracion, 1),
                "arquitectura": config.get("modelo.arquitectura"),
                "semilla": semilla,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    objetivo = 0.9
    if mejor_precision < objetivo:
        logger.warning(
            "La precisión (%.3f) no alcanza el criterio de la fase (%.2f). "
            "Suele faltar dataset: más sesiones y más variedad de condiciones.",
            mejor_precision,
            objetivo,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
