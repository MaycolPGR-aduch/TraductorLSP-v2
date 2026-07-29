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

## Entrenar y exportar (Fase 2)

Las dependencias de entrenamiento van aparte; la app final no las necesita. En
Windows hay que pedir el wheel de CPU explícitamente, porque el de PyPI trae CUDA
y pesa unos 2,4 GB:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch
```

```bash
pip install -e ".[train]"
```

**Antes de entrenar por primera vez**, revisa cómo se están extrayendo las ventanas:

```bash
python scripts/diagnose_windows.py
```

Los parámetros `ventana.umbral_movimiento` y `ventana.fraccion_pico_movimiento`
vienen con valores **provisionales**: se fijaron sin dataset real. El script te
dice si el trazo de cada seña se detecta donde debe y avisa si estás
desaprovechando datos. Ajústalos en el YAML hasta que los avisos desaparezcan.

```bash
python scripts/train.py
```

```bash
python scripts/evaluate.py --matriz models/confusion.png
```

```bash
python scripts/export_onnx.py
```

El export escribe el modelo cuantizado a INT8, comprueba que PyTorch y ONNX
predicen la misma clase, y mide tamaño y latencia contra los límites del YAML.

### Cómo se construyen las muestras

Cada repetición de 3,5 s produce varias ventanas de 2 s. Cuáles son válidas
depende del tipo de seña: de una **estática** sirve cualquier tramo del
sostenimiento; de una **dinámica**, solo las que contienen el trazo casi entero.
Una ventana que solo capta la mano subiendo, etiquetada como la seña, es la causa
típica de predicciones espurias en la app final.

Las ventanas se remuestrean siempre a `frames_por_ventana`, de modo que una
cámara a 25 FPS y otra a 30 produzcan entradas idénticas para el modelo.

### Sobre la evaluación

Los splits son **por sesión completa, nunca por repetición**. Dos repeticiones de
la misma sesión comparten iluminación, ropa, encuadre y el estado del señante ese
día: repartirlas entre train y test hace que el modelo reconozca la sesión en vez
de la seña, y da una precisión que se desploma con usuarios reales. Por eso hace
falta un mínimo de dos sesiones para poder evaluar.

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
