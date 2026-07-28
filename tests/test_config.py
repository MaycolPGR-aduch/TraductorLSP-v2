"""Pruebas de la carga y validación de la configuración."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from senasperu.config import Config, ConfigError, load_config, validate_config


@pytest.fixture(scope="module")
def config() -> Config:
    """Configuración real del proyecto (``config/default.yaml``)."""
    return load_config()


def test_carga_el_yaml_del_proyecto(config: Config) -> None:
    assert config.proyecto.nombre
    assert config.camara.ancho > 0
    assert config.camara.alto > 0


def test_acceso_por_ruta_punteada(config: Config) -> None:
    assert config.get("camara.fps_objetivo") == config.camara.fps_objetivo
    assert config.get("seccion.inexistente", "defecto") == "defecto"


def test_require_falla_si_falta_la_clave(config: Config) -> None:
    with pytest.raises(ConfigError):
        config.require("camara.parametro_inexistente")


def test_atributo_inexistente_lanza_error_descriptivo(config: Config) -> None:
    with pytest.raises(AttributeError, match="camara.parametro_inexistente"):
        _ = config.camara.parametro_inexistente


def test_resolve_path_devuelve_ruta_absoluta(config: Config) -> None:
    ruta = config.resolve_path("dataset.ruta_raiz")
    assert isinstance(ruta, Path)
    assert ruta.is_absolute()


def test_vocabulario_incluye_clase_de_reposo(config: Config) -> None:
    ids = [entrada["id"] for entrada in config.vocabulario]
    assert "no_sena" in ids, "La clase de reposo es obligatoria para evitar falsos positivos."
    assert len(ids) == len(set(ids)), "Hay ids de seña duplicados."


def test_toda_sena_declara_si_es_espejable(config: Config) -> None:
    for entrada in config.vocabulario:
        assert isinstance(entrada["espejable"], bool)


def test_validacion_detecta_secciones_faltantes() -> None:
    with pytest.raises(ConfigError, match="Faltan secciones"):
        validate_config(Config({"proyecto": {"nombre": "x"}}))


def test_validacion_detecta_ids_duplicados(config: Config) -> None:
    datos = config.to_dict()
    datos["vocabulario"].append(dict(datos["vocabulario"][1]))
    with pytest.raises(ConfigError, match="duplicados"):
        validate_config(Config(datos))


def test_validacion_exige_clase_de_reposo(config: Config) -> None:
    datos = config.to_dict()
    datos["vocabulario"] = [e for e in datos["vocabulario"] if e["id"] != "no_sena"]
    with pytest.raises(ConfigError, match="no_sena"):
        validate_config(Config(datos))


def test_archivo_inexistente_lanza_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="No se encontró"):
        load_config(tmp_path / "no_existe.yaml")


def test_yaml_invalido_lanza_config_error(tmp_path: Path) -> None:
    ruta = tmp_path / "roto.yaml"
    ruta.write_text("camara: [sin cerrar\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(ruta)


def test_to_dict_no_comparte_estado(config: Config) -> None:
    copia = config.to_dict()
    copia["camara"]["ancho"] = -1
    assert config.camara.ancho != -1


def test_el_yaml_no_tiene_claves_duplicadas() -> None:
    """Una clave repetida en YAML se pisa en silencio; mejor detectarlo aquí."""
    ruta = Path(__file__).resolve().parents[1] / "config" / "default.yaml"
    texto = ruta.read_text(encoding="utf-8")
    datos = yaml.safe_load(texto)
    claves_top = [
        linea.split(":", 1)[0]
        for linea in texto.splitlines()
        if linea and not linea[0].isspace() and not linea.lstrip().startswith("#") and ":" in linea
    ]
    assert len(claves_top) == len(set(claves_top))
    assert len(claves_top) == len(datos)
