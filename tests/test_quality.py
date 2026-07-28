"""Pruebas del control de calidad de las grabaciones."""

from __future__ import annotations

import pytest

from senasperu.config import load_config
from senasperu.data.quality import QualityChecker


@pytest.fixture
def checker() -> QualityChecker:
    return QualityChecker(
        max_frames_without_hands_pct=20.0,
        min_mean_confidence=0.5,
        min_frames=10,
    )


def test_una_grabacion_buena_se_acepta(checker: QualityChecker) -> None:
    informe = checker.evaluate([2] * 100, [0.9] * 100)
    assert informe.accepted
    assert informe.reasons == ()
    assert informe.without_hands_pct == 0.0


def test_se_rechaza_si_se_pierden_demasiadas_manos(checker: QualityChecker) -> None:
    manos = [0] * 30 + [2] * 70
    informe = checker.evaluate(manos, [0.9] * 100)
    assert not informe.accepted
    assert informe.frames_without_hands == 30
    assert "30 %" in informe.reasons[0]


def test_el_umbral_es_estricto_no_inclusivo(checker: QualityChecker) -> None:
    """Exactamente 20 % pasa; 21 % no. El YAML dice 'más de 20 %'."""
    justo = checker.evaluate([0] * 20 + [2] * 80, [0.9] * 100)
    pasado = checker.evaluate([0] * 21 + [2] * 79, [0.9] * 100)
    assert justo.accepted
    assert not pasado.accepted


def test_se_rechaza_por_confianza_baja(checker: QualityChecker) -> None:
    informe = checker.evaluate([2] * 100, [0.2] * 100)
    assert not informe.accepted
    assert "onfianza" in informe.reasons[0]


def test_se_rechaza_una_grabacion_demasiado_corta(checker: QualityChecker) -> None:
    informe = checker.evaluate([2] * 5, [0.9] * 5)
    assert not informe.accepted
    assert "5 frames" in informe.reasons[0]


def test_sin_frames_se_rechaza_con_mensaje_de_camara(checker: QualityChecker) -> None:
    informe = checker.evaluate([], [])
    assert not informe.accepted
    assert informe.frames == 0
    assert "cámara" in informe.reasons[0]


def test_se_acumulan_todos_los_motivos(checker: QualityChecker) -> None:
    informe = checker.evaluate([0] * 50 + [1] * 50, [0.1] * 100)
    assert len(informe.reasons) == 2


def test_sin_rechazo_automatico_todo_se_acepta() -> None:
    permisivo = QualityChecker(
        max_frames_without_hands_pct=20.0,
        min_mean_confidence=0.5,
        min_frames=10,
        auto_reject=False,
    )
    informe = permisivo.evaluate([0] * 100, [0.0] * 100)
    assert informe.accepted
    assert informe.reasons, "el informe debe explicar los problemas aunque no rechace"


def test_una_sola_mano_es_valida(checker: QualityChecker) -> None:
    """Hay señas de una sola mano: no se puede exigir siempre dos."""
    assert checker.evaluate([1] * 100, [0.9] * 100).accepted


def test_construccion_desde_la_configuracion_del_proyecto() -> None:
    config = load_config()
    verificador = QualityChecker.from_config(config)
    fps = config.camara.fps_objetivo
    duracion = config.grabador.duracion_grabacion_segundos
    completa = int(fps * duracion)

    assert verificador.evaluate([2] * completa, [0.9] * completa).accepted
    assert not verificador.evaluate([2] * 5, [0.9] * 5).accepted


def test_el_resumen_es_legible(checker: QualityChecker) -> None:
    assert checker.evaluate([2] * 100, [0.9] * 100).summary.startswith("Aceptada")
    assert checker.evaluate([], []).summary.startswith("Rechazada")
