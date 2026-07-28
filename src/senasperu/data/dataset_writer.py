"""Escritura del dataset: archivos ``.npz``, video de respaldo y ``metadata.csv``.

Nomenclatura: ``dataset/raw/<seña>/pXX_sYY_rZZ.npz`` (persona, sesión, repetición).
La numeración de repetición la asigna esta clase, nunca el usuario: así no hay
forma de sobrescribir una grabación ni de equivocarse al etiquetar.

Los ``.npz`` son autodescriptivos: además de los landmarks guardan el layout del
vector, los índices faciales usados y las versiones de la app y de MediaPipe. Si
mañana cambia la configuración, los archivos viejos siguen siendo interpretables.
"""

from __future__ import annotations

import csv
import logging
import os
import re
import threading
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from senasperu.config import Config
from senasperu.data.quality import QualityReport
from senasperu.data.recording import RecordingSample

logger = logging.getLogger(__name__)

# pXX_sYY_rZZ
FILE_PATTERN = re.compile(r"^(?P<person>p\d{2})_s(?P<session>\d{2})_r(?P<repetition>\d{2,3})$")
PERSON_PATTERN = re.compile(r"^p\d{2}$")

METADATA_COLUMNS: tuple[str, ...] = (
    "label",
    "persona",
    "sesion",
    "repeticion",
    "fecha",
    "frames",
    "fps",
    "pct_sin_manos",
    "confianza_promedio",
    "iluminacion",
    "distancia",
    "ropa",
    "ruta_npz",
    "ruta_video",
)


@dataclass(frozen=True, slots=True)
class SavedRecording:
    """Datos de una repetición ya escrita en disco."""

    label: str
    person: str
    session: int
    repetition: int
    npz_path: Path
    video_path: Path | None
    frames: int
    fps: float

    @property
    def stem(self) -> str:
        """Nombre del archivo sin extensión (``p01_s03_r07``)."""
        return self.npz_path.stem


class DatasetWriter:
    """Guarda, numera y contabiliza las repeticiones del dataset.

    Todos los métodos son seguros para llamarse desde un hilo distinto al de la
    interfaz; el guardado real ocurre en el hilo trabajador
    (:mod:`senasperu.data.save_worker`) para no congelar la UI.
    """

    def __init__(
        self,
        root: Path,
        metadata_path: Path,
        *,
        save_video: bool,
        app_version: str = "",
    ) -> None:
        """Args:
        root: Carpeta raíz del dataset (``dataset/raw``).
        metadata_path: Ruta del ``metadata.csv``.
        save_video: Si se escribe también el ``.mp4`` de respaldo.
        app_version: Versión de la app, se guarda dentro de cada ``.npz``.
        """
        self._root = Path(root)
        self._metadata_path = Path(metadata_path)
        self._save_video = save_video
        self._app_version = app_version
        self._lock = threading.Lock()
        # (label, person, session) -> cantidad de repeticiones en disco
        self._counts: dict[tuple[str, str, int], int] = defaultdict(int)
        self._last_repetition: dict[tuple[str, str, int], int] = defaultdict(int)
        self.scan()

    @classmethod
    def from_config(cls, config: Config) -> DatasetWriter:
        """Construye el escritor con las rutas y opciones del YAML."""
        from senasperu import __version__

        return cls(
            root=config.resolve_path("dataset.ruta_raiz"),
            metadata_path=config.resolve_path("dataset.metadata_csv"),
            save_video=bool(config.get("grabador.guardar_video_respaldo", False)),
            app_version=__version__,
        )

    # -- Inventario --------------------------------------------------------
    def scan(self) -> None:
        """Recorre el disco y reconstruye los contadores de repeticiones."""
        with self._lock:
            self._counts.clear()
            self._last_repetition.clear()
            if not self._root.is_dir():
                return
            for archivo in self._root.glob("*/*.npz"):
                datos = FILE_PATTERN.match(archivo.stem)
                if datos is None:
                    logger.warning("Archivo con nombre inesperado, se ignora: %s", archivo)
                    continue
                clave = (
                    archivo.parent.name,
                    datos["person"],
                    int(datos["session"]),
                )
                self._counts[clave] += 1
                repeticion = int(datos["repetition"])
                if repeticion > self._last_repetition[clave]:
                    self._last_repetition[clave] = repeticion

    def next_session(self, person: str) -> int:
        """Devuelve la siguiente sesión libre para esa persona (la primera es 1)."""
        _validate_person(person)
        with self._lock:
            sesiones = [clave[2] for clave in self._counts if clave[1] == person]
        return max(sesiones, default=0) + 1

    def count(self, label: str, person: str, session: int | None = None) -> int:
        """Repeticiones grabadas de una seña.

        Args:
            label: Id de la seña.
            person: Persona señante.
            session: Si se indica, cuenta solo esa sesión; si no, todas.
        """
        with self._lock:
            if session is not None:
                return self._counts.get((label, person, session), 0)
            return sum(
                cantidad
                for (etiqueta, persona, _sesion), cantidad in self._counts.items()
                if etiqueta == label and persona == person
            )

    def counts_by_label(self, person: str, session: int | None = None) -> dict[str, int]:
        """Repeticiones por seña, para alimentar el contador de la interfaz."""
        resultado: dict[str, int] = defaultdict(int)
        with self._lock:
            for (etiqueta, persona, sesion), cantidad in self._counts.items():
                if persona != person:
                    continue
                if session is not None and sesion != session:
                    continue
                resultado[etiqueta] += cantidad
        return dict(resultado)

    # -- Escritura ---------------------------------------------------------
    def save(
        self,
        sample: RecordingSample,
        *,
        person: str,
        session: int,
        report: QualityReport,
        conditions: dict[str, str] | None = None,
    ) -> SavedRecording:
        """Escribe una repetición en disco y anota su fila en ``metadata.csv``.

        Args:
            sample: Muestra grabada.
            person: Persona señante (``p01``).
            session: Número de sesión.
            report: Informe de calidad, se resume en el CSV.
            conditions: Condiciones de grabación (iluminación, distancia, ropa).

        Returns:
            Los datos de la grabación guardada.
        """
        _validate_person(person)
        if not sample.label:
            raise ValueError("La muestra no tiene seña asignada; no se puede guardar.")

        carpeta = self._root / sample.label
        carpeta.mkdir(parents=True, exist_ok=True)

        with self._lock:
            clave = (sample.label, person, session)
            repeticion = self._last_repetition[clave] + 1
            self._last_repetition[clave] = repeticion
            self._counts[clave] += 1

        nombre = f"{person}_s{session:02d}_r{repeticion:02d}"
        ruta_npz = carpeta / f"{nombre}.npz"
        self._write_npz(ruta_npz, sample, person=person, session=session, repetition=repeticion)

        ruta_video: Path | None = None
        if self._save_video and sample.video_frames:
            ruta_video = carpeta / f"{nombre}.mp4"
            if not _write_video(ruta_video, sample.video_frames, sample.fps):
                ruta_video = None

        guardada = SavedRecording(
            label=sample.label,
            person=person,
            session=session,
            repetition=repeticion,
            npz_path=ruta_npz,
            video_path=ruta_video,
            frames=sample.frames,
            fps=sample.fps,
        )
        self._append_metadata(guardada, report, conditions or {})
        logger.info(
            "Guardada %s (%s frames a %.1f FPS)", ruta_npz.name, sample.frames, sample.fps
        )
        return guardada

    def discard(self, saved: SavedRecording) -> bool:
        """Borra una repetición guardada y su fila del ``metadata.csv``.

        Returns:
            ``True`` si se borró algo.
        """
        borrado = False
        for ruta in (saved.npz_path, saved.video_path):
            if ruta is not None and ruta.is_file():
                ruta.unlink()
                borrado = True

        with self._lock:
            clave = (saved.label, saved.person, saved.session)
            if self._counts.get(clave):
                self._counts[clave] -= 1
            # Se libera el número para que la siguiente grabación lo reutilice.
            if self._last_repetition.get(clave) == saved.repetition:
                self._last_repetition[clave] = saved.repetition - 1

        self._remove_metadata_row(saved)
        if borrado:
            logger.info("Descartada la grabación %s", saved.stem)
        return borrado

    # -- Detalles de escritura --------------------------------------------
    def _write_npz(
        self,
        path: Path,
        sample: RecordingSample,
        *,
        person: str,
        session: int,
        repetition: int,
    ) -> None:
        """Escribe el ``.npz`` de forma atómica (archivo temporal + rename)."""
        temporal = path.with_suffix(".npz.tmp")
        with temporal.open("wb") as archivo:
            np.savez_compressed(
                archivo,
                landmarks=sample.landmarks,
                confidence=sample.confidence,
                hands_per_frame=sample.hands_per_frame,
                fps=np.float32(sample.fps),
                label=sample.label,
                person=person,
                session=np.int16(session),
                repetition=np.int16(repetition),
                layout_names=np.array(sample.layout.names),
                layout_points=np.array([b.points for b in sample.layout.blocks], dtype=np.int32),
                layout_coords=np.array([b.coords for b in sample.layout.blocks], dtype=np.int32),
                created_at=datetime.now().isoformat(timespec="seconds"),
                app_version=self._app_version,
            )
        os.replace(temporal, path)

    def _append_metadata(
        self,
        saved: SavedRecording,
        report: QualityReport,
        conditions: dict[str, str],
    ) -> None:
        """Agrega una fila al ``metadata.csv``, creándolo con cabecera si hace falta."""
        self._metadata_path.parent.mkdir(parents=True, exist_ok=True)
        nuevo = not self._metadata_path.exists()
        fila = {
            "label": saved.label,
            "persona": saved.person,
            "sesion": saved.session,
            "repeticion": saved.repetition,
            "fecha": datetime.now().isoformat(timespec="seconds"),
            "frames": saved.frames,
            "fps": f"{saved.fps:.2f}",
            "pct_sin_manos": f"{report.without_hands_pct:.1f}",
            "confianza_promedio": f"{report.mean_confidence:.3f}",
            "iluminacion": conditions.get("iluminacion", ""),
            "distancia": conditions.get("distancia", ""),
            "ropa": conditions.get("ropa", ""),
            "ruta_npz": _relative(saved.npz_path, self._metadata_path.parent),
            "ruta_video": _relative(saved.video_path, self._metadata_path.parent),
        }
        with self._lock, self._metadata_path.open("a", encoding="utf-8", newline="") as archivo:
            escritor = csv.DictWriter(archivo, fieldnames=METADATA_COLUMNS)
            if nuevo:
                escritor.writeheader()
            escritor.writerow(fila)

    def _remove_metadata_row(self, saved: SavedRecording) -> None:
        """Reescribe el ``metadata.csv`` sin la fila de esa grabación."""
        if not self._metadata_path.is_file():
            return
        objetivo = _relative(saved.npz_path, self._metadata_path.parent)
        with self._lock:
            with self._metadata_path.open("r", encoding="utf-8", newline="") as archivo:
                lector = csv.DictReader(archivo)
                columnas = lector.fieldnames or list(METADATA_COLUMNS)
                filas = [fila for fila in lector if fila.get("ruta_npz") != objetivo]
            temporal = self._metadata_path.with_suffix(".csv.tmp")
            with temporal.open("w", encoding="utf-8", newline="") as archivo:
                escritor = csv.DictWriter(archivo, fieldnames=columnas)
                escritor.writeheader()
                escritor.writerows(filas)
            os.replace(temporal, self._metadata_path)


def _validate_person(person: str) -> None:
    """Verifica el formato ``pXX`` del identificador de persona."""
    if not PERSON_PATTERN.match(person):
        raise ValueError(
            f"El identificador de persona '{person}' no tiene el formato pXX (por ejemplo p01)."
        )


def _relative(path: Path | None, base: Path) -> str:
    """Ruta relativa a ``base`` si es posible; si no, la absoluta."""
    if path is None:
        return ""
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


def _write_video(path: Path, frames: tuple[np.ndarray, ...], fps: float) -> bool:
    """Escribe el video de respaldo. Devuelve ``False`` si no se pudo."""
    import cv2

    if not frames:
        return False
    alto, ancho = frames[0].shape[:2]
    fps_video = fps if fps > 1.0 else 30.0
    escritor = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps_video, (ancho, alto))
    if not escritor.isOpened():
        logger.warning("No se pudo crear el video de respaldo %s; se guardan solo landmarks.", path)
        return False
    try:
        for frame in frames:
            escritor.write(frame)
    finally:
        escritor.release()
    return True
