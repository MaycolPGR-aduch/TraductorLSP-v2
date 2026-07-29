"""Export a ONNX, cuantización INT8 y benchmark de latencia en CPU (Fase 2).

Verifica los tres criterios de aceptación de la fase de una sola pasada:
tamaño <10 MB, latencia <50 ms por ventana y equivalencia numérica entre el
modelo PyTorch y el ONNX exportado.

Uso::

    python scripts/export_onnx.py
    python scripts/export_onnx.py --sin-cuantizar
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ / "src") not in sys.path:
    sys.path.insert(0, str(RAIZ / "src"))

import torch  # noqa: E402

from senasperu.config import ConfigError, load_config  # noqa: E402
from senasperu.logging_setup import setup_logging  # noqa: E402
from senasperu.model.architecture import build_model  # noqa: E402

MB = 1024 * 1024


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define y procesa los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="Exporta el modelo a ONNX y lo cuantiza.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--salida", type=Path, default=None)
    parser.add_argument(
        "--sin-cuantizar", action="store_true", help="Exporta en float32, sin INT8."
    )
    parser.add_argument("--repeticiones", type=int, default=100, help="Iteraciones del benchmark.")
    return parser.parse_args(argv)


def benchmark(ruta_onnx: Path, ejemplo: np.ndarray, threads: int, repeticiones: int) -> dict:
    """Mide la latencia por ventana en CPU."""
    from senasperu.model.inference import SignClassifier

    clasificador = SignClassifier(ruta_onnx, threads=threads)
    for _ in range(10):  # calentamiento: la primera inferencia siempre es lenta
        clasificador.predict(ejemplo)

    tiempos = []
    for _ in range(repeticiones):
        inicio = time.perf_counter()
        clasificador.predict(ejemplo)
        tiempos.append((time.perf_counter() - inicio) * 1000.0)

    tiempos_ordenados = sorted(tiempos)
    return {
        "media_ms": float(np.mean(tiempos)),
        "mediana_ms": float(np.median(tiempos)),
        "p95_ms": float(tiempos_ordenados[int(0.95 * len(tiempos_ordenados)) - 1]),
        "max_ms": float(max(tiempos)),
    }


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada del export."""
    args = parse_args(argv)
    try:
        config = load_config(args.config)
    except ConfigError as error:
        print(f"Error de configuración: {error}", file=sys.stderr)
        return 2
    logger = setup_logging(config)

    ruta_checkpoint = Path(args.checkpoint) if args.checkpoint else RAIZ / "models" / "senasperu.pt"
    if not ruta_checkpoint.is_file():
        logger.error(
            "No se encontró el checkpoint %s. Entrena primero con 'python scripts/train.py'.",
            ruta_checkpoint,
        )
        return 1

    checkpoint = torch.load(ruta_checkpoint, map_location="cpu", weights_only=False)
    datos = config.to_dict()
    datos["modelo"]["arquitectura"] = checkpoint.get(
        "architecture", config.get("modelo.arquitectura")
    )
    from senasperu.config import Config

    config_modelo = Config(datos)

    modelo = build_model(config_modelo, checkpoint["input_size"], checkpoint["num_classes"])
    modelo.load_state_dict(checkpoint["state_dict"])
    modelo.eval()

    ejemplo = torch.zeros(
        1, checkpoint["frames_per_window"], checkpoint["input_size"], dtype=torch.float32
    )
    ruta_final = Path(args.salida) if args.salida else config.resolve_path(
        "inferencia.ruta_modelo_onnx"
    )
    ruta_final.parent.mkdir(parents=True, exist_ok=True)
    ruta_float = ruta_final.with_name(ruta_final.stem.replace("_int8", "") + "_fp32.onnx")

    torch.onnx.export(
        modelo,
        (ejemplo,),
        str(ruta_float),
        input_names=["ventana"],
        output_names=["logits"],
        # El lote es dinámico; frames y features son fijos a propósito, para que
        # el modelo falle temprano si la configuración de features cambia.
        dynamic_axes={"ventana": {0: "lote"}, "logits": {0: "lote"}},
        opset_version=17,
        # Exportador clásico (TorchScript). El nuevo, basado en dynamo, genera
        # un grafo que la cuantización dinámica de ONNX Runtime no consigue
        # convertir. Si algún día se elimina, habrá que revisar este paso.
        dynamo=False,
    )
    logger.info("Exportado a ONNX: %s (%.2f MB)", ruta_float, ruta_float.stat().st_size / MB)

    if args.sin_cuantizar:
        ruta_final = ruta_float
    else:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(
            model_input=str(ruta_float),
            model_output=str(ruta_final),
            weight_type=QuantType.QInt8,
        )
        logger.info(
            "Cuantizado a INT8: %s (%.2f MB)", ruta_final, ruta_final.stat().st_size / MB
        )

    # Equivalencia numérica: la cuantización no debe cambiar la clase predicha.
    entrada = np.random.default_rng(0).normal(
        0.0, 0.5, size=(8, checkpoint["frames_per_window"], checkpoint["input_size"])
    ).astype(np.float32)
    with torch.no_grad():
        referencia = modelo(torch.from_numpy(entrada)).numpy()

    from senasperu.model.inference import SignClassifier

    clasificador = SignClassifier(ruta_final, threads=int(config.get("inferencia.hilos_onnx", 2)))
    coincidencias = sum(
        int(clasificador.predict(entrada[i]).class_index == int(np.argmax(referencia[i])))
        for i in range(entrada.shape[0])
    )
    logger.info(
        "Coincidencia de clase PyTorch vs ONNX: %s/%s", coincidencias, entrada.shape[0]
    )

    medidas = benchmark(
        ruta_final,
        entrada[0],
        int(config.get("inferencia.hilos_onnx", 2)),
        int(args.repeticiones),
    )
    tamano_mb = ruta_final.stat().st_size / MB
    limite_mb = float(config.get("modelo.tamano_max_mb", 10))
    limite_ms = float(config.get("modelo.latencia_max_ms", 50))

    logger.info(
        "Latencia por ventana: media %.1f ms | mediana %.1f ms | p95 %.1f ms | máx %.1f ms",
        medidas["media_ms"],
        medidas["mediana_ms"],
        medidas["p95_ms"],
        medidas["max_ms"],
    )

    print("\n--- Criterios de aceptación de Fase 2 ---")
    tamano_ok = tamano_mb < limite_mb
    latencia_ok = medidas["p95_ms"] < limite_ms
    print(f"  Tamaño   : {tamano_mb:6.2f} MB  (límite {limite_mb:.0f} MB)   {_marca(tamano_ok)}")
    print(f"  Latencia : {medidas['p95_ms']:6.1f} ms  (límite {limite_ms:.0f} ms, p95) {_marca(latencia_ok)}")
    print(f"  Equivalencia PyTorch/ONNX: {coincidencias}/{entrada.shape[0]}")
    print(f"\nModelo listo en: {ruta_final}")
    return 0 if (tamano_ok and latencia_ok) else 1


def _marca(ok: bool) -> str:
    return "OK" if ok else "NO CUMPLE"


if __name__ == "__main__":
    raise SystemExit(main())
