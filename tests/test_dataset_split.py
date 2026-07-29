"""Pruebas del split por sesión.

Es la prueba que protege la honestidad de toda la evaluación: si una sesión se
reparte entre entrenamiento y prueba, la precisión medida deja de significar nada.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from senasperu.data.dataset import Recording, scan_recordings, split_by_session


def grabaciones(sesiones: int, por_sesion: int = 4, etiquetas=("hola", "agua")) -> list[Recording]:
    salida: list[Recording] = []
    for sesion in range(1, sesiones + 1):
        for repeticion in range(1, por_sesion + 1):
            etiqueta = etiquetas[repeticion % len(etiquetas)]
            salida.append(
                Recording(
                    path=Path(f"raw/{etiqueta}/p01_s{sesion:02d}_r{repeticion:02d}.npz"),
                    label=etiqueta,
                    person="p01",
                    session=sesion,
                    repetition=repeticion,
                )
            )
    return salida


def test_ninguna_sesion_aparece_en_ambos_lados() -> None:
    train, test = split_by_session(grabaciones(6), test_ratio=0.3, seed=1)
    sesiones_train = {g.session_key for g in train}
    sesiones_test = {g.session_key for g in test}
    assert sesiones_train.isdisjoint(sesiones_test)


def test_no_se_pierde_ni_se_duplica_ninguna_grabacion() -> None:
    todas = grabaciones(6)
    train, test = split_by_session(todas, test_ratio=0.3, seed=1)
    assert len(train) + len(test) == len(todas)
    assert {g.path for g in train} | {g.path for g in test} == {g.path for g in todas}


def test_con_una_sola_sesion_no_hay_split_honesto() -> None:
    with pytest.raises(ValueError, match="una sola sesión"):
        split_by_session(grabaciones(1), test_ratio=0.3, seed=1)


def test_es_reproducible_con_la_misma_semilla() -> None:
    primero = split_by_session(grabaciones(8), test_ratio=0.25, seed=42)[1]
    segundo = split_by_session(grabaciones(8), test_ratio=0.25, seed=42)[1]
    assert [g.path for g in primero] == [g.path for g in segundo]


def test_ninguna_clase_se_queda_sin_ejemplos_de_entrenamiento() -> None:
    """Apartar una sesión no puede dejar una seña sin datos para aprender."""
    todas = grabaciones(4)
    train, _ = split_by_session(todas, test_ratio=0.5, seed=3)
    assert {g.label for g in train} == {g.label for g in todas}


def test_la_proporcion_de_prueba_se_respeta_aproximadamente() -> None:
    todas = grabaciones(10, por_sesion=5)
    _, test = split_by_session(todas, test_ratio=0.3, seed=5)
    proporcion = len(test) / len(todas)
    assert 0.15 <= proporcion <= 0.45, f"proporción real {proporcion:.2f}"


def test_scan_ignora_archivos_con_nombre_invalido(tmp_path: Path) -> None:
    carpeta = tmp_path / "hola"
    carpeta.mkdir(parents=True)
    (carpeta / "p01_s01_r01.npz").write_bytes(b"")
    (carpeta / "grabacion_suelta.npz").write_bytes(b"")

    encontradas = scan_recordings(tmp_path)
    assert len(encontradas) == 1
    assert encontradas[0].person == "p01"
    assert encontradas[0].session == 1
    assert encontradas[0].label == "hola"


def test_scan_en_carpeta_inexistente_devuelve_lista_vacia(tmp_path: Path) -> None:
    assert scan_recordings(tmp_path / "no_existe") == []
