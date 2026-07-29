"""Diagnóstico de la detección de trazo y la extracción de ventanas.

**Ejecuta esto antes de entrenar por primera vez.** Los parámetros
``ventana.umbral_movimiento`` y ``ventana.fraccion_pico_movimiento`` vienen con
valores provisionales: se fijaron sin dataset real y hay que ajustarlos con tus
grabaciones. Este script muestra, por cada repetición, la velocidad de las
muñecas, el tramo que se detecta como trazo y cuántas ventanas se extraen.

Qué mirar:

- El **trazo** de una seña dinámica debería caer más o menos en el centro de la
  toma y durar bastante más que un puñado de frames. Si sale pegado al borde o
  dura 5 frames, el umbral está mal.
- Si muchas grabaciones producen **una sola ventana**, el filtro está siendo
  demasiado estricto: baja ``cobertura_minima_trazo`` o el umbral.
- Si el **p90** de la velocidad es parecido a la mediana, esa seña casi no se
  mueve: probablemente sea estática y esté marcada como dinámica en el YAML.

Uso::

    python scripts/diagnose_windows.py
    python scripts/diagnose_windows.py --sena hola
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ / "src") not in sys.path:
    sys.path.insert(0, str(RAIZ / "src"))

from senasperu.config import ConfigError, load_config  # noqa: E402
from senasperu.data.dataset import scan_recordings  # noqa: E402
from senasperu.features.normalize import LandmarkNormalizer  # noqa: E402
from senasperu.features.vector import layout_from_config  # noqa: E402
from senasperu.features.windows import WindowExtractor  # noqa: E402
from senasperu.vocabulary import find_sign, load_vocabulary  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define y procesa los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="Diagnostica la extracción de ventanas.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--sena", type=str, default=None, help="Filtra por id de seña.")
    parser.add_argument("--limite", type=int, default=40, help="Máximo de filas a mostrar.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada del diagnóstico."""
    args = parse_args(argv)
    try:
        config = load_config(args.config)
    except ConfigError as error:
        print(f"Error de configuración: {error}", file=sys.stderr)
        return 2

    raiz = Path(args.dataset) if args.dataset else config.resolve_path("dataset.ruta_raiz")
    grabaciones = scan_recordings(raiz)
    if args.sena:
        grabaciones = [g for g in grabaciones if g.label == args.sena]
    if not grabaciones:
        print(f"No hay grabaciones en {raiz}.")
        return 1

    layout = layout_from_config(config)
    extractor = WindowExtractor.from_config(config, layout)
    vocabulario = load_vocabulary(config)

    print(f"Grabaciones: {len(grabaciones)}   Raíz: {raiz}")
    print(f"umbral_movimiento = {config.get('ventana.umbral_movimiento')}   ", end="")
    print(f"fraccion_pico_movimiento = {config.get('ventana.fraccion_pico_movimiento')}   ", end="")
    print(f"cobertura_minima_trazo = {config.get('ventana.cobertura_minima_trazo')}\n")
    print(
        f"{'archivo':<22}{'seña':<12}{'tipo':<10}{'frames':>7}{'p90':>7}{'mediana':>9}"
        f"   {'trazo':<14}{'ventanas':>9}"
    )

    ventanas_por_grabacion: list[int] = []
    trazos_sospechosos = 0
    for grabacion in grabaciones[: args.limite]:
        with np.load(grabacion.path, allow_pickle=False) as datos:
            landmarks = datos["landmarks"]
            fps = float(datos["fps"])
            ancho = int(datos["frame_width"]) if "frame_width" in datos else 0
            alto = int(datos["frame_height"]) if "frame_height" in datos else 0

        aspecto = (ancho / alto) if ancho and alto else None
        secuencia = LandmarkNormalizer.from_config(
            config, layout, aspect_ratio=aspecto
        ).normalize(landmarks)
        velocidad = extractor.wrist_speed(secuencia, fps)
        sign = find_sign(vocabulario, grabacion.label)
        dinamica = sign is not None and not sign.is_static
        trazo = extractor.active_segment(secuencia, fps) if dinamica else None
        ventanas = extractor.extract(secuencia, fps, dynamic=dinamica)
        ventanas_por_grabacion.append(len(ventanas))

        if dinamica:
            largo_trazo = (trazo[1] - trazo[0]) if trazo else 0
            if trazo is None or largo_trazo < 0.2 * secuencia.frames:
                trazos_sospechosos += 1

        print(
            f"{grabacion.path.name:<22}{grabacion.label:<12}"
            f"{('dinámica' if dinamica else 'estática'):<10}{secuencia.frames:>7}"
            f"{np.percentile(velocidad, 90) if velocidad.size else 0:>7.2f}"
            f"{np.median(velocidad) if velocidad.size else 0:>9.2f}   "
            f"{(f'{trazo[0]}-{trazo[1]}' if trazo else '-'):<14}{len(ventanas):>9}"
        )

    reparto = Counter(ventanas_por_grabacion)
    print(f"\nVentanas por grabación: {dict(sorted(reparto.items()))}")
    print(f"Total de ventanas      : {sum(ventanas_por_grabacion)}")
    if trazos_sospechosos:
        print(
            f"\nNOTA: {trazos_sospechosos} grabaciones dinámicas se trataron como "
            "estáticas porque no se detectó un trazo con forma plausible. No se "
            "pierde nada: todas sus ventanas se usan. Es lo esperable en señas "
            "oscilantes, como un saludo, donde la mano se mueve durante toda la "
            "toma. Solo merece revisión si esperabas un trazo claro y aislado: "
            "en ese caso baja 'ventana.umbral_movimiento'."
        )
    if reparto.get(1, 0) > len(ventanas_por_grabacion) / 2:
        print(
            "\nAVISO: más de la mitad de las grabaciones producen una sola ventana. "
            "Estás desaprovechando datos: baja 'ventana.cobertura_minima_trazo'."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
