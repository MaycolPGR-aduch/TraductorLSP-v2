"""Pruebas de los medidores de FPS y latencia."""

from __future__ import annotations

import pytest

from senasperu.utils import FpsMeter, LatencyMeter


def test_fps_sin_datos_es_cero() -> None:
    medidor = FpsMeter()
    assert medidor.fps == 0.0
    medidor.tick(0.0)
    assert medidor.fps == 0.0


def test_fps_con_intervalos_regulares() -> None:
    medidor = FpsMeter(window=10)
    for i in range(11):
        medidor.tick(i * 0.04)  # 25 FPS exactos
    assert medidor.fps == pytest.approx(25.0)


def test_fps_usa_solo_la_ventana_reciente() -> None:
    """Un arranque lento no debe penalizar la medición actual."""
    medidor = FpsMeter(window=5)
    for i in range(3):
        medidor.tick(i * 1.0)  # 1 FPS
    base = 3.0
    for i in range(1, 11):
        medidor.tick(base + i * 0.02)  # 50 FPS
    assert medidor.fps > 45.0


def test_fps_reset() -> None:
    medidor = FpsMeter()
    medidor.tick(0.0)
    medidor.tick(0.1)
    medidor.reset()
    assert medidor.fps == 0.0


def test_latencia_promedia_en_milisegundos() -> None:
    medidor = LatencyMeter(window=3)
    medidor.add(0.010)
    medidor.add(0.020)
    assert medidor.milliseconds == 15.0


def test_latencia_sin_muestras() -> None:
    assert LatencyMeter().milliseconds == 0.0
