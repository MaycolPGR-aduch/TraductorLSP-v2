"""Pruebas de la cola con descarte del elemento más viejo.

Esta cola es la que garantiza la regla "antes descartar frames que acumular
retraso", así que su comportamiento se verifica sin cámara ni hilos de Qt.
"""

from __future__ import annotations

import threading
import time

import pytest

from senasperu.capture.frame_queue import DropOldestQueue


def test_maxsize_invalido() -> None:
    with pytest.raises(ValueError):
        DropOldestQueue(0)


def test_orden_fifo_sin_desbordar() -> None:
    cola: DropOldestQueue[int] = DropOldestQueue(3)
    for valor in (1, 2, 3):
        assert cola.put(valor) is False
    assert [cola.get(timeout=0) for _ in range(3)] == [1, 2, 3]
    assert cola.dropped == 0


def test_descarta_el_mas_viejo_al_llenarse() -> None:
    cola: DropOldestQueue[int] = DropOldestQueue(2)
    cola.put(1)
    cola.put(2)
    assert cola.put(3) is True  # descarta el 1
    assert cola.qsize() == 2
    assert cola.get(timeout=0) == 2
    assert cola.get(timeout=0) == 3
    assert cola.dropped == 1


def test_get_devuelve_none_al_expirar_el_tiempo() -> None:
    cola: DropOldestQueue[int] = DropOldestQueue(1)
    inicio = time.perf_counter()
    assert cola.get(timeout=0.05) is None
    assert time.perf_counter() - inicio < 1.0


def test_get_latest_descarta_los_intermedios() -> None:
    cola: DropOldestQueue[int] = DropOldestQueue(5)
    for valor in range(5):
        cola.put(valor)
    assert cola.get_latest(timeout=0) == 4
    assert cola.empty()


def test_get_latest_sin_datos() -> None:
    cola: DropOldestQueue[int] = DropOldestQueue(2)
    assert cola.get_latest(timeout=0) is None


def test_clear_vacia_sin_contar_descartes() -> None:
    cola: DropOldestQueue[int] = DropOldestQueue(3)
    cola.put(1)
    cola.put(2)
    cola.clear()
    assert cola.empty()
    assert cola.dropped == 0


def test_el_productor_nunca_se_bloquea() -> None:
    """Un productor rápido con consumidor lento no debe bloquearse ni crecer."""
    cola: DropOldestQueue[int] = DropOldestQueue(2)
    inicio = time.perf_counter()
    for valor in range(10_000):
        cola.put(valor)
    duracion = time.perf_counter() - inicio

    assert duracion < 2.0, "put() se está bloqueando o es demasiado lento"
    assert cola.qsize() <= 2, "la cola creció por encima de su capacidad"
    assert cola.dropped == 9_998
    assert cola.get(timeout=0) == 9_998  # solo sobreviven los más recientes


def test_uso_concurrente_productor_consumidor() -> None:
    """Con un hilo productor y uno consumidor, el consumidor ve valores crecientes."""
    cola: DropOldestQueue[int] = DropOldestQueue(2)
    recibidos: list[int] = []
    fin = threading.Event()

    def producir() -> None:
        for valor in range(2_000):
            cola.put(valor)
        fin.set()

    productor = threading.Thread(target=producir, daemon=True)
    productor.start()
    while not fin.is_set() or not cola.empty():
        item = cola.get(timeout=0.05)
        if item is not None:
            recibidos.append(item)
    productor.join(timeout=2.0)

    assert not productor.is_alive()
    assert recibidos, "el consumidor no recibió ningún elemento"
    assert recibidos == sorted(recibidos), "los elementos llegaron desordenados"
    assert len(recibidos) + cola.dropped == 2_000
