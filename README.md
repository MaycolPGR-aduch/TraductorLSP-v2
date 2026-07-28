# SeñasPerú

Traductor de **Lengua de Señas Peruana (LSP)** a texto y voz en tiempo real, con webcam.
100 % offline, solo CPU, solo software libre.

> Estado: **Fase 1 — Grabador de dataset**. Listo para grabar señas.

## Requisitos

- **Python 3.11 exactamente** (64 bits). No sirve 3.12 ni superior: el proyecto
  usa `mediapipe==0.10.21`, la última versión con la API `solutions.holistic`,
  y esa versión solo publica wheels hasta cp311 en Windows.
- Una webcam.
- Windows, Linux o macOS. Probado en Windows 11.

### Por qué MediaPipe está clavado en 0.10.21

Google eliminó las soluciones legacy (`mp.solutions.*`) a partir de la 0.10.30.
La 0.10.21 trae Holistic con sus modelos `.tflite` **dentro de la wheel**, así que
no hay nada que descargar y la app queda 100 % offline sin pasos extra.

Migrar algún día a la API Tasks (`HolisticLandmarker`) obliga a re-extraer el
dataset completo desde los videos de respaldo, porque los landmarks no serían
idénticos a los ya grabados. Por eso `grabador.guardar_video_respaldo` está en
`true`: es el seguro que hace esa migración posible.

## Instalación

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

En Linux/macOS el activado es `source .venv/bin/activate`.

Las dependencias de entrenamiento (PyTorch) **no** se instalan por defecto: la
aplicación final no las necesita. Cuando llegue la Fase 2:

```bash
pip install -e ".[train]"
```

## Probar la Fase 0

```bash
python scripts/smoke_test.py
```

Se abre una ventana con la cámara, el esqueleto de landmarks superpuesto y un panel
de rendimiento. Criterios de aceptación de la fase:

| Métrica | Objetivo |
|---|---|
| FPS en pantalla | ≥ 25 |
| CPU | < 60 % |
| Memoria tras 10 min | estable (sin crecimiento sostenido) |

Opciones útiles:

```bash
python scripts/smoke_test.py --camara 1
```

```bash
python scripts/smoke_test.py --video ruta/al/video.mp4
```

```bash
python scripts/smoke_test.py --sin-landmarks
```

Si los FPS quedan por debajo de 25, las palancas están todas en
`config/default.yaml`, en este orden: `mediapipe.model_complexity: 0`,
`mediapipe.usar_rostro: false`, y bajar `camara.ancho`/`camara.alto`.

## Grabar el dataset (Fase 1)

```bash
python scripts/record_dataset.py
```

Para grabar a otra persona señante (el YAML define la que viene por defecto):

```bash
python scripts/record_dataset.py --persona p02
```

### Cómo se graba

Todo se maneja con el teclado, sin tocar el mouse:

| Tecla | Acción |
|---|---|
| `Espacio` | Grabar: 2 s de cuenta regresiva y luego 3,5 s de captura |
| `←` `→` | Cambiar de seña |
| `R` | Repetir la última seña grabada |
| `D` | Descartar la última repetición guardada |
| `Esc` | Salir |

El borde de la vista de cámara indica el estado: gris en reposo, ámbar en la
cuenta regresiva y rojo mientras graba.

Al terminar cada toma se evalúa la calidad en el momento. Si MediaPipe perdió las
manos en más del 20 % de los frames o la confianza promedio quedó baja, la
repetición **se rechaza y no se guarda**: se explica el motivo y se regraba con `R`.

### Contra los errores de etiquetado

- La seña sale siempre de la lista del vocabulario; no hay ningún campo de texto libre.
- La seña se congela al iniciar la cuenta regresiva: cambiarla a mitad de una toma es imposible.
- El nombre del archivo y la numeración de repetición los asigna el programa, nunca tú.

### Consejos de encuadre

MediaPipe Holistic deduce la posición de las manos a partir de la pose. Si te
acercas tanto que el torso sale del cuadro, pierde el cuerpo y con él las manos,
y todas las tomas se rechazan. Ubícate a **1,5 m aproximadamente, con hombros y
torso visibles**, y con luz de frente.

### Qué se guarda

```
dataset/
  raw/<seña>/pXX_sYY_rZZ.npz     landmarks crudos + confianza + fps + layout
  raw/<seña>/pXX_sYY_rZZ.mp4     respaldo de video (sin el esqueleto dibujado)
  metadata.csv                   una fila por repetición, con las condiciones
```

Los landmarks se guardan **sin normalizar**, y las partes no detectadas van como
`NaN` (un cero es una coordenada válida y se confundiría con "ausente"). Así se
puede cambiar el criterio de normalización en Fase 2 sin regrabar nada. Cada
`.npz` incluye además el layout del vector, para seguir siendo interpretable si
la configuración cambia.

## Pruebas

```bash
pytest
```

Las pruebas no necesitan webcam: cubren configuración, cola de frames y medidores.

## Estructura

```
config/default.yaml     Todos los parámetros del sistema (ningún número mágico en el código)
src/senasperu/
  capture/              Fuentes de frames, cola con descarte, hilo de captura
  features/             MediaPipe Holistic, landmarks, dibujo del esqueleto
  model/                (Fase 2) arquitectura, export ONNX, inferencia
  stabilize/            (Fase 3) umbral, votación, debouncing
  tts/                  (Fase 3) voz offline con Piper
  ui/                   PySide6: smoke test y grabador
  data/                 Dataset: control de calidad, escritura .npz y metadata
scripts/                Puntos de entrada ejecutables
tests/                  pytest (sin cámara)
dataset/  models/  logs/    Ignorados por git
```

## Decisiones de arquitectura

- **Hilos separados**: captura → cola → procesamiento → cola → UI. Las colas son
  cortas y descartan el frame más viejo: preferimos perder frames antes que
  acumular latencia.
- **Los hilos de trabajo no dependen de Qt** (`threading.Thread` puro), para poder
  probarlos sin GUI. Un puente en `ui/pipeline_bridge.py` los conecta a las
  señales de Qt.
- **La UI no toca OpenCV ni MediaPipe.** El dibujo del esqueleto ocurre en el hilo
  de procesamiento.
- **Se espeja el frame en la captura**, no solo en pantalla, para que lo que el
  usuario ve, lo que MediaPipe procesa y lo que se graba coincidan.

## Licencia

GPL-3.0-or-later (compatible con las dependencias de código abierto usadas).
