"""Pruebas del vocabulario leído de la configuración."""

from __future__ import annotations

from senasperu.config import load_config
from senasperu.vocabulary import REST_SIGN_ID, find_sign, load_vocabulary


def test_se_carga_el_vocabulario_completo() -> None:
    config = load_config()
    vocabulario = load_vocabulary(config)
    assert len(vocabulario) == len(config.vocabulario)
    assert all(sign.id and sign.glosa for sign in vocabulario)


def test_los_indices_son_la_posicion_en_el_yaml() -> None:
    """El índice es la clase del modelo: debe seguir el orden del archivo."""
    vocabulario = load_vocabulary(load_config())
    assert [sign.index for sign in vocabulario] == list(range(len(vocabulario)))


def test_la_clase_de_reposo_existe_y_se_reconoce() -> None:
    vocabulario = load_vocabulary(load_config())
    reposo = find_sign(vocabulario, REST_SIGN_ID)
    assert reposo is not None
    assert reposo.is_rest
    assert reposo.text == "", "la clase de reposo no debe producir texto"


def test_solo_la_clase_de_reposo_es_reposo() -> None:
    vocabulario = load_vocabulary(load_config())
    assert sum(1 for sign in vocabulario if sign.is_rest) == 1


def test_buscar_una_sena_inexistente_devuelve_none() -> None:
    vocabulario = load_vocabulary(load_config())
    assert find_sign(vocabulario, "no_existe") is None


def test_todas_las_senas_declaran_espejado() -> None:
    for sign in load_vocabulary(load_config()):
        assert isinstance(sign.mirrorable, bool)
