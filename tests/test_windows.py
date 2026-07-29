"""Pruebas de la extracción de ventanas de entrenamiento."""

from __future__ import annotations

import numpy as np

from senasperu.features.landmarks import POSE_LEFT_WRIST, POSE_RIGHT_WRIST
from senasperu.features.normalize import NormalizedSequence
from senasperu.features.vector import BLOCK_POSE, build_layout
from senasperu.features.windows import WindowExtractor

LAYOUT = build_layout(face_points=0)
FEATURES = LAYOUT.size + len(LAYOUT.blocks)
FPS = 30.0


def extractor(**cambios) -> WindowExtractor:
    opciones = dict(
        window_seconds=2.0,
        frames_per_window=48,
        stride_frames=8,
        min_stroke_coverage=0.8,
        motion_threshold=0.35,
        smoothing_frames=5,
        min_valid_ratio=0.7,
    )
    opciones.update(cambios)
    return WindowExtractor(LAYOUT, **opciones)


def secuencia(frames: int, *, movimiento: tuple[int, int] | None = None) -> NormalizedSequence:
    """Secuencia normalizada; opcionalmente con las muñecas moviéndose en un tramo."""
    datos = np.zeros((frames, FEATURES), dtype=np.float32)
    bloque = LAYOUT.block(BLOCK_POSE)
    pose = np.zeros((frames, bloque.points, bloque.coords), dtype=np.float32)
    if movimiento is not None:
        inicio, fin = movimiento
        recorrido = np.zeros(frames, dtype=np.float32)
        recorrido[inicio:fin] = np.linspace(0.0, 1.5, fin - inicio, dtype=np.float32)
        recorrido[fin:] = 1.5
        pose[:, POSE_LEFT_WRIST, 0] = recorrido
        pose[:, POSE_RIGHT_WRIST, 0] = recorrido
    # reshape(frames, -1) es ambiguo con frames = 0: se indica el largo exacto.
    datos[:, bloque.slice] = pose.reshape(frames, bloque.points * bloque.coords)
    return NormalizedSequence(features=datos, valid=np.ones(frames, dtype=bool))


def test_una_sena_estatica_da_varias_ventanas() -> None:
    ventanas = extractor().extract(secuencia(105), FPS, dynamic=False)
    assert len(ventanas) > 1
    assert all(v.features.shape == (48, FEATURES) for v in ventanas)


def test_todas_las_ventanas_se_remuestrean_al_mismo_largo() -> None:
    """Una cámara a 25 FPS y otra a 30 deben producir entradas idénticas."""
    a = extractor().extract(secuencia(105), 30.0, dynamic=False)
    b = extractor().extract(secuencia(88), 25.0, dynamic=False)
    assert a and b
    assert a[0].features.shape == b[0].features.shape == (48, FEATURES)


def test_en_senas_dinamicas_solo_valen_las_ventanas_con_el_trazo() -> None:
    """Una ventana que solo capta la preparación no debe etiquetarse como la seña."""
    datos = secuencia(150, movimiento=(60, 90))
    dinamicas = extractor().extract(datos, FPS, dynamic=True)
    estaticas = extractor().extract(datos, FPS, dynamic=False)

    assert dinamicas, "debería quedar al menos una ventana con el trazo"
    assert len(dinamicas) < len(estaticas), "el filtro por trazo no descartó nada"
    trazo = extractor().active_segment(datos, FPS)
    assert trazo is not None
    for ventana in dinamicas:
        fin = ventana.start_frame + int(2.0 * FPS)
        solape = min(fin, trazo[1]) - max(ventana.start_frame, trazo[0])
        assert solape / (trazo[1] - trazo[0]) >= 0.8


def test_el_trazo_se_localiza_donde_esta_el_movimiento() -> None:
    trazo = extractor().active_segment(secuencia(150, movimiento=(60, 90)), FPS)
    assert trazo is not None
    inicio, fin = trazo
    assert 50 <= inicio <= 70, f"el trazo empieza en {inicio}"
    assert 80 <= fin <= 100, f"el trazo termina en {fin}"


def test_sin_trazo_una_dinamica_se_trata_como_estatica() -> None:
    """Si no hay trazo distinguible, todas las ventanas valen: no se pierde nada."""
    quieta = secuencia(105)
    assert extractor().active_segment(quieta, FPS) is None
    assert len(extractor().extract(quieta, FPS, dynamic=True)) == len(
        extractor().extract(quieta, FPS, dynamic=False)
    )


def test_un_movimiento_continuo_no_se_toma_como_trazo() -> None:
    """Un saludo oscilante mueve la mano toda la toma; filtrar por 'trazo' ahí
    descartaría ventanas perfectamente buenas."""
    continuo = secuencia(105, movimiento=(0, 105))
    assert extractor().active_segment(continuo, FPS) is None
    assert len(extractor().extract(continuo, FPS, dynamic=True)) > 1


def test_un_destello_de_ruido_no_se_toma_como_trazo() -> None:
    """Un tramo demasiado corto es jitter del landmark, no una seña."""
    datos = secuencia(105, movimiento=(50, 53))
    assert extractor().active_segment(datos, FPS) is None


def test_se_descartan_las_ventanas_con_demasiado_torso_perdido() -> None:
    datos = secuencia(105)
    invalidos = np.ones(105, dtype=bool)
    invalidos[:70] = False  # las primeras ventanas quedan casi sin torso
    parcial = NormalizedSequence(features=datos.features, valid=invalidos)

    completas = extractor().extract(datos, FPS, dynamic=False)
    filtradas = extractor().extract(parcial, FPS, dynamic=False)
    assert len(filtradas) < len(completas)


def test_una_grabacion_mas_corta_que_la_ventana_se_usa_entera() -> None:
    ventanas = extractor().extract(secuencia(20), FPS, dynamic=False)
    assert len(ventanas) == 1
    assert ventanas[0].features.shape == (48, FEATURES)


def test_una_secuencia_vacia_no_da_ventanas() -> None:
    assert extractor().extract(secuencia(0), FPS, dynamic=False) == []
    assert extractor().extract(secuencia(50), 0.0, dynamic=False) == []
