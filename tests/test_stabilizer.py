"""Pruebas de la capa de estabilización.

Aquí se decide si la app escribe palabras que nadie señó, así que se prueban
tanto los casos buenos como los ruidosos.
"""

from __future__ import annotations

import pytest

from senasperu.config import load_config
from senasperu.stabilize.stabilizer import Stabilizer

REPOSO = 0
HOLA = 1
GRACIAS = 2


def estabilizador(**cambios) -> Stabilizer:
    opciones = dict(
        rest_index=REPOSO,
        confidence_threshold=0.7,
        vote_windows=5,
        debounce_seconds=0.5,
        require_rest_between_repeats=True,
    )
    opciones.update(cambios)
    return Stabilizer(**opciones)


def alimentar(
    stab: Stabilizer, clase: int, *, confianza: float = 0.9, veces: int = 1, inicio: float = 0.0,
    paso: float = 0.1,
) -> list:
    """Envía varias predicciones seguidas y devuelve los estados."""
    return [
        stab.update(clase, confianza, inicio + i * paso) for i in range(veces)
    ]


def test_una_sena_sostenida_se_confirma_una_sola_vez() -> None:
    stab = estabilizador()
    estados = alimentar(stab, HOLA, veces=20)
    confirmaciones = [e.confirmed for e in estados if e.confirmed is not None]
    assert confirmaciones == [HOLA], "una seña sostenida debe escribirse una vez"


def test_no_se_confirma_antes_del_debounce() -> None:
    stab = estabilizador(debounce_seconds=0.5)
    estados = alimentar(stab, HOLA, veces=5, paso=0.05)  # 0,25 s en total
    assert all(e.confirmed is None for e in estados)


def test_las_predicciones_por_debajo_del_umbral_no_votan() -> None:
    """Este es el filtro que evita traducir cualquier movimiento cotidiano."""
    stab = estabilizador(confidence_threshold=0.7)
    estados = alimentar(stab, HOLA, confianza=0.5, veces=30)
    assert all(e.confirmed is None for e in estados)
    assert all(e.candidate is None for e in estados)


def test_un_frame_raro_aislado_no_produce_traduccion() -> None:
    """El caso clásico: una ventana suelta con una predicción equivocada."""
    stab = estabilizador()
    alimentar(stab, REPOSO, veces=10)
    intruso = stab.update(GRACIAS, 0.95, 1.1)
    siguientes = alimentar(stab, REPOSO, veces=10, inicio=1.2)
    assert intruso.confirmed is None
    assert all(e.confirmed is None for e in siguientes)


def test_el_reposo_nunca_se_confirma_como_traduccion() -> None:
    stab = estabilizador()
    estados = alimentar(stab, REPOSO, veces=40)
    assert all(e.confirmed is None for e in estados)
    assert estados[-1].at_rest


def test_una_sena_no_se_repite_sin_pasar_por_reposo() -> None:
    stab = estabilizador()
    primera = alimentar(stab, HOLA, veces=20)
    assert [e.confirmed for e in primera].count(HOLA) == 1

    # Sigue señando HOLA sin descansar: no debe volver a escribirse.
    segunda = alimentar(stab, HOLA, veces=20, inicio=3.0)
    assert all(e.confirmed is None for e in segunda)


def test_tras_pasar_por_reposo_la_misma_sena_se_puede_repetir() -> None:
    stab = estabilizador()
    alimentar(stab, HOLA, veces=20)
    alimentar(stab, REPOSO, veces=20, inicio=3.0)
    repetida = alimentar(stab, HOLA, veces=20, inicio=6.0)
    assert [e.confirmed for e in repetida].count(HOLA) == 1


def test_sin_exigir_reposo_la_sena_puede_repetirse() -> None:
    stab = estabilizador(require_rest_between_repeats=False)
    estados = alimentar(stab, HOLA, veces=40)
    assert [e.confirmed for e in estados].count(HOLA) >= 2


def test_dos_senas_distintas_seguidas_se_confirman_ambas() -> None:
    stab = estabilizador()
    primera = alimentar(stab, HOLA, veces=15)
    segunda = alimentar(stab, GRACIAS, veces=15, inicio=2.0)
    assert [e.confirmed for e in primera].count(HOLA) == 1
    assert [e.confirmed for e in segunda].count(GRACIAS) == 1


def test_sin_mayoria_clara_no_hay_candidato() -> None:
    """Predicciones alternando entre dos clases no deben confirmar nada."""
    stab = estabilizador()
    estados = []
    for i in range(40):
        clase = HOLA if i % 2 == 0 else GRACIAS
        estados.append(stab.update(clase, 0.9, i * 0.1))
    assert all(e.confirmed is None for e in estados)


def test_el_progreso_avanza_con_el_tiempo() -> None:
    stab = estabilizador(debounce_seconds=1.0)
    estados = alimentar(stab, HOLA, veces=8, paso=0.1)
    progresos = [e.progress for e in estados]
    assert progresos[-1] > progresos[2] > 0.0
    assert all(0.0 <= p <= 1.0 for p in progresos)


def test_reset_olvida_el_historial() -> None:
    stab = estabilizador()
    alimentar(stab, HOLA, veces=20)
    stab.reset()
    repetida = alimentar(stab, HOLA, veces=20, inicio=10.0)
    assert [e.confirmed for e in repetida].count(HOLA) == 1


def test_dos_minutos_de_reposo_no_producen_ni_una_traduccion() -> None:
    """Criterio de aceptación de la fase, con predicciones ruidosas de verdad."""
    import random

    rng = random.Random(7)
    stab = estabilizador()
    confirmaciones = 0
    for i in range(3600):  # 2 minutos a 30 predicciones por segundo
        # Casi siempre reposo, con predicciones sueltas poco confiables.
        if rng.random() < 0.05:
            clase, confianza = rng.choice([HOLA, GRACIAS]), rng.uniform(0.3, 0.69)
        else:
            clase, confianza = REPOSO, rng.uniform(0.75, 0.99)
        if stab.update(clase, confianza, i / 30.0).confirmed is not None:
            confirmaciones += 1
    assert confirmaciones == 0, f"hubo {confirmaciones} traducciones espurias en reposo"


def test_construccion_desde_la_configuracion_del_proyecto() -> None:
    config = load_config()
    stab = Stabilizer.from_config(config, rest_index=REPOSO)
    debounce = float(config.estabilizacion.debounce_segundos)
    estados = alimentar(stab, HOLA, veces=int(debounce / 0.05) + 10, paso=0.05)
    assert [e.confirmed for e in estados].count(HOLA) == 1


@pytest.mark.parametrize("umbral", [0.5, 0.7, 0.9])
def test_el_umbral_se_respeta(umbral: float) -> None:
    stab = estabilizador(confidence_threshold=umbral)
    justo_debajo = alimentar(stab, HOLA, confianza=umbral - 0.01, veces=20)
    assert all(e.confirmed is None for e in justo_debajo)

    stab = estabilizador(confidence_threshold=umbral)
    justo_encima = alimentar(stab, HOLA, confianza=umbral + 0.01, veces=20)
    assert [e.confirmed for e in justo_encima].count(HOLA) == 1
