"""Pruebas de la escritura del dataset: nomenclatura, numeración y metadata."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from senasperu.data.dataset_writer import DatasetWriter
from senasperu.data.quality import QualityReport
from senasperu.data.recording import RecordingSample
from senasperu.features.vector import build_layout

LAYOUT = build_layout(face_points=2)


def muestra(label: str = "hola", frames: int = 40, *, con_video: bool = False) -> RecordingSample:
    """Muestra sintética, sin cámara."""
    imagenes = (
        tuple(np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(frames)) if con_video else None
    )
    return RecordingSample(
        label=label,
        landmarks=np.zeros((frames, LAYOUT.size), dtype=np.float32),
        confidence=np.full(frames, 0.9, dtype=np.float32),
        hands_per_frame=np.full(frames, 2, dtype=np.int8),
        fps=30.0,
        layout=LAYOUT,
        video_frames=imagenes,
    )


def informe(frames: int = 40) -> QualityReport:
    return QualityReport(
        accepted=True,
        frames=frames,
        frames_without_hands=0,
        without_hands_pct=0.0,
        mean_confidence=0.9,
    )


@pytest.fixture
def writer(tmp_path: Path) -> DatasetWriter:
    return DatasetWriter(
        root=tmp_path / "raw",
        metadata_path=tmp_path / "metadata.csv",
        save_video=False,
        app_version="test",
    )


def test_nomenclatura_y_carpeta(writer: DatasetWriter) -> None:
    guardada = writer.save(muestra(), person="p01", session=1, report=informe())
    assert guardada.npz_path.name == "p01_s01_r01.npz"
    assert guardada.npz_path.parent.name == "hola"
    assert guardada.npz_path.is_file()


def test_la_numeracion_avanza_sola(writer: DatasetWriter) -> None:
    nombres = [
        writer.save(muestra(), person="p01", session=1, report=informe()).npz_path.name
        for _ in range(3)
    ]
    assert nombres == ["p01_s01_r01.npz", "p01_s01_r02.npz", "p01_s01_r03.npz"]


def test_cada_sesion_numera_desde_uno(writer: DatasetWriter) -> None:
    writer.save(muestra(), person="p01", session=1, report=informe())
    segunda = writer.save(muestra(), person="p01", session=2, report=informe())
    assert segunda.npz_path.name == "p01_s02_r01.npz"


def test_personas_distintas_no_se_pisan(writer: DatasetWriter) -> None:
    primera = writer.save(muestra(), person="p01", session=1, report=informe())
    segunda = writer.save(muestra(), person="p02", session=1, report=informe())
    assert primera.npz_path != segunda.npz_path
    assert segunda.npz_path.name == "p02_s01_r01.npz"


def test_persona_con_formato_invalido(writer: DatasetWriter) -> None:
    with pytest.raises(ValueError, match="pXX"):
        writer.save(muestra(), person="maycol", session=1, report=informe())


def test_el_npz_conserva_los_datos_y_se_describe_a_si_mismo(writer: DatasetWriter) -> None:
    guardada = writer.save(muestra(frames=33), person="p03", session=7, report=informe(33))
    with np.load(guardada.npz_path, allow_pickle=False) as datos:
        assert datos["landmarks"].shape == (33, LAYOUT.size)
        assert datos["confidence"].shape == (33,)
        assert datos["hands_per_frame"].shape == (33,)
        assert str(datos["label"]) == "hola"
        assert str(datos["person"]) == "p03"
        assert int(datos["session"]) == 7
        assert int(datos["repetition"]) == 1
        assert float(datos["fps"]) == pytest.approx(30.0)
        # Autodescriptivo: el layout viaja con los datos.
        assert list(datos["layout_names"]) == list(LAYOUT.names)
        assert list(datos["layout_points"]) == [b.points for b in LAYOUT.blocks]
        assert str(datos["app_version"]) == "test"


def test_el_metadata_csv_lleva_cabecera_y_una_fila_por_grabacion(writer: DatasetWriter) -> None:
    writer.save(muestra("gracias"), person="p01", session=1, report=informe())
    writer.save(muestra("hola"), person="p01", session=1, report=informe())

    with (writer._metadata_path).open(encoding="utf-8", newline="") as archivo:
        filas = list(csv.DictReader(archivo))

    assert len(filas) == 2
    assert filas[0]["label"] == "gracias"
    assert filas[0]["persona"] == "p01"
    assert filas[0]["ruta_npz"] == "raw/gracias/p01_s01_r01.npz"


def test_las_condiciones_se_registran(writer: DatasetWriter) -> None:
    writer.save(
        muestra(),
        person="p01",
        session=1,
        report=informe(),
        conditions={"iluminacion": "baja", "distancia": "lejos", "ropa": "clara"},
    )
    with (writer._metadata_path).open(encoding="utf-8", newline="") as archivo:
        fila = next(csv.DictReader(archivo))
    assert (fila["iluminacion"], fila["distancia"], fila["ropa"]) == ("baja", "lejos", "clara")


def test_descartar_borra_archivo_fila_y_libera_el_numero(writer: DatasetWriter) -> None:
    primera = writer.save(muestra(), person="p01", session=1, report=informe())
    segunda = writer.save(muestra(), person="p01", session=1, report=informe())

    assert writer.discard(segunda)
    assert not segunda.npz_path.exists()
    assert primera.npz_path.exists()

    with (writer._metadata_path).open(encoding="utf-8", newline="") as archivo:
        filas = list(csv.DictReader(archivo))
    assert len(filas) == 1
    assert filas[0]["repeticion"] == "1"

    # El número descartado se reutiliza: no quedan huecos en la numeración.
    tercera = writer.save(muestra(), person="p01", session=1, report=informe())
    assert tercera.npz_path.name == "p01_s01_r02.npz"


def test_contadores_por_sena_y_sesion(writer: DatasetWriter) -> None:
    writer.save(muestra("hola"), person="p01", session=1, report=informe())
    writer.save(muestra("hola"), person="p01", session=2, report=informe())
    writer.save(muestra("agua"), person="p01", session=2, report=informe())

    assert writer.count("hola", "p01") == 2
    assert writer.count("hola", "p01", 1) == 1
    assert writer.counts_by_label("p01") == {"hola": 2, "agua": 1}
    assert writer.counts_by_label("p01", 2) == {"hola": 1, "agua": 1}


def test_next_session_continua_donde_quedo(writer: DatasetWriter) -> None:
    assert writer.next_session("p01") == 1
    writer.save(muestra(), person="p01", session=1, report=informe())
    assert writer.next_session("p01") == 2
    assert writer.next_session("p02") == 1


def test_scan_reconstruye_los_contadores_desde_el_disco(tmp_path: Path) -> None:
    primero = DatasetWriter(
        root=tmp_path / "raw", metadata_path=tmp_path / "metadata.csv", save_video=False
    )
    primero.save(muestra(), person="p01", session=3, report=informe())

    segundo = DatasetWriter(
        root=tmp_path / "raw", metadata_path=tmp_path / "metadata.csv", save_video=False
    )
    assert segundo.count("hola", "p01") == 1
    assert segundo.next_session("p01") == 4
    # La siguiente repetición continúa la numeración, no la pisa.
    assert segundo.save(muestra(), person="p01", session=3, report=informe()).repetition == 2


def test_el_video_de_respaldo_se_escribe_cuando_esta_activo(tmp_path: Path) -> None:
    writer = DatasetWriter(
        root=tmp_path / "raw", metadata_path=tmp_path / "metadata.csv", save_video=True
    )
    guardada = writer.save(muestra(con_video=True), person="p01", session=1, report=informe())

    assert guardada.video_path is not None
    assert guardada.video_path.is_file()
    assert guardada.video_path.stat().st_size > 0

    with (writer._metadata_path).open(encoding="utf-8", newline="") as archivo:
        fila = next(csv.DictReader(archivo))
    assert fila["ruta_video"].endswith("p01_s01_r01.mp4")


def test_sin_respaldo_no_se_escribe_video(writer: DatasetWriter) -> None:
    guardada = writer.save(muestra(con_video=True), person="p01", session=1, report=informe())
    assert guardada.video_path is None
