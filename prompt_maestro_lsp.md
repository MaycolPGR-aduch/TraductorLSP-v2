# CLAUDE.md — Traductor de Lengua de Señas Peruana (LSP)

## Contexto del proyecto

Estás construyendo **SeñasPerú** (nombre provisional): una aplicación de escritorio que traduce Lengua de Señas Peruana a texto y voz en tiempo real usando una webcam. El objetivo es reducir brechas de comunicación para personas sordas en Perú. No es un prototipo académico: debe ser estable, usable por personas reales, y ejecutable en hardware modesto.

**Usuario objetivo del software:** personas sordas señantes de LSP y sus interlocutores oyentes.
**Desarrollador:** una sola persona, que también grabará el dataset inicial por sí misma (más señantes se sumarán después — la estructura de datos debe anticiparlo).

## Restricciones no negociables

1. **CPU-only en inferencia.** Todo debe correr fluido en una laptop Core i5 de hace 5 años con 8 GB de RAM, sin GPU. El entrenamiento puede asumir Google Colab, pero la app final es solo CPU.
2. **100% offline.** Ninguna funcionalidad de la app puede depender de internet.
3. **Solo código abierto.** Sin servicios de pago, sin APIs cloud, sin licencias propietarias.
4. **Estabilidad ante todo.** La app no debe congelarse, acumular latencia ni "parpadear" predicciones. Prefiere descartar frames antes que encolar retraso.
5. **Idioma:** interfaz, mensajes, logs de usuario y documentación en español. Nombres de variables/funciones en inglés (convención estándar de código).

## Stack tecnológico (fijo, no lo cambies sin preguntar)

| Componente | Herramienta | Notas |
|---|---|---|
| Lenguaje | Python 3.11+ | |
| Captura de video | OpenCV (`opencv-python`) | Hilo dedicado |
| Extracción de features | MediaPipe Holistic | Landmarks de manos (21×2), pose (33), rostro reducido. NO se procesa video crudo con CNNs |
| Entrenamiento | PyTorch | Solo en scripts de training, no es dependencia de la app final |
| Inferencia | ONNX Runtime | Modelo exportado y cuantizado a INT8 |
| TTS | Piper TTS | Voz en español, offline |
| GUI | PySide6 (Qt) | Licencia LGPL |
| Empaquetado | PyInstaller | Al final del proyecto |
| Datos | NumPy `.npz` + `metadata.csv` | Se guardan landmarks, no video (video opcional como respaldo) |

## Arquitectura de la aplicación en tiempo real

Tres hilos comunicados por colas (`queue.Queue` con `maxsize` pequeño y política de descarte del frame más viejo):

```
[Hilo captura] → cola_frames → [Hilo inferencia] → cola_resultados → [Hilo UI (Qt main thread)]
```

- **Hilo de captura:** lee la webcam a ~30 FPS, publica frames. Si la cola está llena, descarta el frame viejo y pone el nuevo (nunca bloquea).
- **Hilo de inferencia:** MediaPipe → normalización → buffer de ventana deslizante → modelo ONNX → capa de estabilización.
- **UI:** recibe resultados vía señales de Qt (`Signal`/`Slot`), nunca toca OpenCV/MediaPipe directamente. La UI jamás se bloquea.

### Capa de estabilización (obligatoria)

Sobre las predicciones crudas del modelo, en este orden:
1. **Umbral de confianza:** predicciones bajo 0.7 se descartan (umbral configurable).
2. **Votación por mayoría** sobre las últimas N ventanas (N configurable, default 5).
3. **Debouncing:** una seña se confirma solo si domina durante ≥0.5 s; una seña confirmada no se repite hasta pasar por reposo.
4. **Clase explícita "no-seña/reposo"** en el modelo: manos quietas o movimientos cotidianos nunca deben producir traducciones.

### Normalización de landmarks (crítica para generalizar)

- Centrar coordenadas respecto a un punto de referencia estable (ej. punto medio de hombros).
- Escalar por la distancia entre hombros (invarianza a distancia de cámara y tamaño corporal).
- Manejar frames con detección perdida: interpolar huecos cortos (≤3 frames), marcar inválidos los largos.

## Estructura del repositorio

```
senasperu/
  pyproject.toml
  README.md
  config/
    default.yaml          # TODOS los parámetros configurables viven aquí
  src/senasperu/
    capture/              # webcam, hilo de captura
    features/             # MediaPipe, normalización, buffer de ventanas
    model/                # arquitectura PyTorch, export a ONNX, wrapper de inferencia
    stabilize/            # umbral, votación, debouncing
    tts/                  # integración Piper
    ui/                   # PySide6: app principal y grabador de dataset
    data/                 # dataset: guardado .npz, metadata, validación de calidad
  scripts/
    record_dataset.py     # lanza el grabador
    train.py              # entrenamiento (Colab-friendly)
    export_onnx.py        # export + cuantización INT8
    evaluate.py           # evaluación con splits por sesión
    run_app.py            # app de traducción en tiempo real
  tests/
  dataset/                # (gitignored) raw/<seña>/pXX_sYY_rZZ.npz + metadata.csv
  models/                 # (gitignored) checkpoints y .onnx
```

## Formato del dataset

- Archivo por repetición: `pXX_sYY_rZZ.npz` (persona, sesión, repetición).
- Contenido del `.npz`: `landmarks` (frames × features), `confidence` (por frame), `fps`, `label`.
- `metadata.csv`: label, persona, sesión, fecha, condiciones (iluminación, distancia, ropa), ruta.
- **Control de calidad automático:** si MediaPipe pierde las manos en >20% de los frames, la grabación se rechaza en el momento y se pide regrabar.
- **Splits de evaluación SIEMPRE por sesión completa** (nunca mezclar repeticiones de una misma sesión entre train y test). Documenta esto en `evaluate.py`.

## Modelo

- Entrada: secuencia de vectores de landmarks normalizados (ventana deslizante, ~2 s).
- Arquitectura: Transformer encoder pequeño (2-4 capas) o BiLSTM como baseline. Objetivo: <10 MB cuantizado, inferencia <50 ms por ventana en CPU.
- Salida fase 1: clasificación de señas aisladas + clase "no-seña".
- Data augmentation sobre landmarks: rotación leve, escalado, ruido gaussiano, jitter temporal (velocidad ±20%), espejado horizontal solo si la seña lo permite (configurable por seña).

## Fases de construcción (en este orden)

### Fase 0 — Esqueleto
Estructura del repo, `pyproject.toml`, config YAML, logging, y un smoke test: abrir webcam, correr MediaPipe, dibujar landmarks en una ventana PySide6 a ≥25 FPS. **No avances hasta que esto corra fluido.**

### Fase 1 — Grabador de dataset (prioridad máxima)
App PySide6 con: vista de cámara con landmarks superpuestos, selector de seña (lista desde config), grabación con cuenta regresiva de 2 s + captura de 3-4 s, guardado automático con nomenclatura correcta, contador de repeticiones por seña, rechazo automático por baja calidad de detección, y campo de persona/sesión. Atajos de teclado para grabar sin tocar el mouse.

### Fase 2 — Entrenamiento y export
Dataset loader con augmentation, script de entrenamiento reproducible (semillas fijas), evaluación con splits por sesión, matriz de confusión, export a ONNX + cuantización, y benchmark de latencia en CPU.

### Fase 3 — App de traducción en tiempo real
La arquitectura de 3 hilos completa, capa de estabilización, UI con: vista de cámara, texto traducido acumulándose en pantalla (fuente grande, alto contraste), botón para reproducir con voz (Piper), indicador de confianza, y modo "historial de conversación".

### Fase 4 — Empaquetado y robustez
PyInstaller, manejo de errores de cámara desconectada, selección de cámara, primera ejecución guiada.

## Convenciones de código

- Type hints en todas las firmas públicas. Docstrings en español.
- Configuración centralizada: ningún número mágico en el código; todo parámetro (umbrales, tamaños de ventana, FPS, rutas) sale de `config/default.yaml`.
- Tests con pytest para: normalización de landmarks, buffer de ventanas, capa de estabilización (estas tres son lógica pura y fáciles de testear sin cámara).
- Cada módulo debe poder probarse sin webcam (inyección de dependencias: la fuente de frames es una interfaz, con implementación de cámara real y de archivo de video para tests).
- Commits pequeños y descriptivos. No mezcles fases.

## Criterios de aceptación por fase

- **Fase 0:** ≥25 FPS con landmarks dibujados, CPU <60% en un i5 antiguo, sin fugas de memoria en 10 min.
- **Fase 1:** grabar 15 repeticiones de una seña toma <4 min; cero errores de etiquetado posibles por diseño; grabaciones malas se rechazan en el momento.
- **Fase 2:** precisión >90% en split por sesión con ~20 señas; latencia de inferencia <50 ms/ventana en CPU; modelo <10 MB.
- **Fase 3:** latencia seña→texto <1 s percibida; cero predicciones espurias en 2 min de reposo frente a la cámara; la UI nunca se congela.

## Cómo trabajar conmigo

- Empieza SIEMPRE confirmando en qué fase estamos y proponiendo un plan corto antes de escribir código.
- Si una decisión técnica contradice las restricciones no negociables, detente y pregunta.
- Al terminar cada bloque de trabajo, dime cómo probarlo manualmente en 1-2 pasos.
- Prioriza que cada fase termine en algo ejecutable y demostrable, no en código a medias de varias fases.
