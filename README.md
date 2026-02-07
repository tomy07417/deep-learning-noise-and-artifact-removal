# 🎥 Eliminación de Manchas Fijas en Videos usando U-Net

Un sistema de deep learning para remover manchas estáticas en videos (como las causadas por suciedad en la lente) aprovechando el movimiento temporal como información de reconstrucción.

---

## 📋 Tabla de Contenidos

1. [Descripción del Problema](#descripción-del-problema)
2. [Idea Principal](#idea-principal)
3. [Arquitectura del Modelo](#arquitectura-del-modelo)
4. [Dataset](#dataset)
5. [Funcionalidades](#funcionalidades)
6. [Estructura del Proyecto](#estructura-del-proyecto)
7. [Instalación y Uso](#instalación-y-uso)
8. [Resultados](#resultados)
9. [Mejoras Futuras](#mejoras-futuras)

---

## 📌 Descripción del Problema

### El Desafío

Cuando una lente tiene una mancha fija (polvo, humedad, rayón, etc.), esa imperfección aparece en **todos los frames** del video. Sin embargo, el contenido detrás de la mancha **sí se mueve**.

**Ejemplo visual:**
```
Frame 1: Dígito en posición A + mancha en (x, y)
Frame 2: Dígito en posición B + mancha en (x, y)  ← misma mancha, contenido diferente
Frame 3: Dígito en posición C + mancha en (x, y)  ← podemos inferir qué hay detrás
```

### Por Qué es Posible Resolver Esto

Si analizamos varios frames consecutivos, la **información temporal** nos permite reconstruir lo que está ocultado:

- La mancha siempre está en el **mismo píxel (x, y)** en todos los frames
- El contenido detrás de la mancha **cambia** de un frame al siguiente
- Con suficientes frames, tenemos "ventanas" donde diferentes partes del contenido se ven alrededor de la mancha

Este es el núcleo de la solución: **el movimiento del contenido proporciona la información que necesitamos para reconstruir las áreas ocultas**.

---

## 💡 Idea Principal

El modelo toma un **frame corrupto** (con mancha) y genera un **frame restaurado** (sin mancha). Lo inteligente es que:

1. **El modelo ve un solo frame** → debe aprender que la mancha es anómala
2. **El temporal smoothing** → garantiza coherencia entre frames consecutivos
3. **En conjunto**: El video reconstruido es fluido y visualmente coherente

### Pipeline Completo

```
Video Original → Agregar Mancha → Entrenar Modelo → Video Reconstruido
    (limpio)     (sintetizada)      (U-Net + TS)    (sin mancha)
```

---

## 🏗️ Arquitectura del Modelo

### U-Net: Arquitectura Encoder-Decoder

La **U-Net** es una arquitectura diseñada específicamente para tareas de **restauración y segmentación pixel-a-pixel**.

#### Estructura General

```
Entrada: (B, C_in, H, W)  → (B, 3, 256, 256)

ENCODER (Contracción)
├── Initial Conv      → (B, 64, 256, 256)
├── Down Conv 1       → (B, 128, 128, 128)   [MaxPool 2x2]
├── Down Conv 2       → (B, 256, 64, 64)     [MaxPool 2x2]
├── Down Conv 3       → (B, 512, 32, 32)     [MaxPool 2x2]
└── Down Conv 4       → (B, 1024, 16, 16)    [MaxPool 2x2]

BOTTLENECK (Cuello de botella)
└── (B, 1024, 16, 16)

DECODER (Expansión con conexiones residuales)
├── Up Conv 1 + Skip from Down 3   → (B, 512, 32, 32)   [Upsample 2x2]
├── Up Conv 2 + Skip from Down 2   → (B, 256, 64, 64)   [Upsample 2x2]
├── Up Conv 3 + Skip from Down 1   → (B, 128, 128, 128) [Upsample 2x2]
└── Up Conv 4 + Skip from Initial  → (B, 64, 256, 256)  [Upsample 2x2]

Final Conv
└── Salida: (B, C_out, H, W)  → (B, 3, 256, 256)
```

#### Componentes Clave

**1. DoubleConv: Bloque de Convolución Doble**
```python
Conv(3x3) → BatchNorm → ReLU
    ↓
Conv(3x3) → BatchNorm → ReLU
```
Permite aprender características más complejas con menos parámetros.

**2. Skip Connections (Conexiones Residuales)**
```
Encoder: x₁ ──────────────────────┐
          ↓                       │
         x₂ ─────────────┐        │
          ↓              │        │
         x₃ ──────┐      │        │
          ↓       │      │        │
        [Bottleneck]     │        │
          ↓       │      │        │
         u₁ ◄─────┘      │        │
          ↓              │        │
         u₂ ◄────────────┘        │
          ↓                       │
         u₃ ◄─────────────────────┘
```

Las **skip connections** permiten que información de alta resolución del encoder llegue directamente al decoder, evitando pérdida de detalles.

**3. DownConv: Codificación (compresión)**
- MaxPool 2x2 para reducir dimensiones
- DoubleConv para extraer características
- Reduce espacialmente, aumenta canales

**4. UpConv: Decodificación (expansión)**
- Upsample 2x2 (bicúbico) para aumentar resolución
- Conv 1x1 para ajustar canales
- Concatenación (cat) con skip connections
- DoubleConv para procesar la fusión

#### Configuración del Modelo

```python
CHANNELS_IN = 3        # RGB
CHANNELS = 64          # Base channels

Modelo:
├── Initial Conv:  3 → 64
├── Down Conv 1:   64 → 128
├── Down Conv 2:   128 → 256
├── Down Conv 3:   256 → 512
├── Down Conv 4:   512 → 1024
├── Up Conv 1:     1024 → 512
├── Up Conv 2:     512 → 256
├── Up Conv 3:     256 → 128
├── Up Conv 4:     128 → 64
└── Final Conv:    64 → 3 (RGB)

Parámetros totales: ~13.4M
```

---

### Temporal Smoothing: Coherencia en el Tiempo

Después de que la U-Net procesa cada frame por separado, aplicamos **temporal smoothing** para garantizar coherencia entre frames consecutivos.

#### El Problema: Flickering

Aunque todos los frames sean casi idénticos:
- Frame t → Red output = [127.3, 128.1, 126.8]
- Frame t+1 → Red output = [127.5, 129.2, 127.1]
- Frame t+2 → Red output = [126.8, 127.9, 126.5]

Las pequeñas variaciones crean un efecto de **parpadeo visual (flickering)** incómodo.

#### La Solución: Promediado Temporal Ponderado

```
smoothed[t] = strength × frame[t] + 
              ((1 - strength) / 2) × frame[t-1] + 
              ((1 - strength) / 2) × frame[t+1]
```

Con `strength = 0.6`:
```
smoothed[t] = 0.6 × frame[t] + 0.2 × frame[t-1] + 0.2 × frame[t+1]
```

**Ventajas:**
- Mantiene el frame central como referencia (60%)
- Promedia con vecinos para suavidad (40% distribuido)
- Los bordes (frame 0 y último) se mantienen sin suavizar
- Reduce ruido y variaciones abruptas

**Resultado:** Videos más fluidos y coherentes visualmente.

---

## 📊 Dataset

### Fuente: MNIST Animado

**Dataset original:** [Captioned Moving MNIST - Medium Version](https://www.kaggle.com/datasets/yichengs/captioned-moving-mnist-dataset-medium-version)

**Características:**
- Dígitos manuscritos (0-9) moviendo aleatoriamente en un canvas
- Movimiento consistente y predecible
- Fondo simple y uniforme
- Ideal para prototiping de modelos de visión temporal

### Generación del Dataset de Entrenamiento

El proceso está automatizado en `generacion_dataset.ipynb`:

#### Paso 1: Corte de Videos
```python
Duración original: ~60 segundos
Duración corte: 15 segundos (360 frames @ 24fps)
Razón: Datos más manejables, reduce almacenamiento
```

#### Paso 2: Síntesis de Manchas

**Función `generate_circular_stain()`:**
```python
def generate_circular_stain(h, w, radius=25, opacity=0.7, hardness=0.8):
    # 1. Centro aleatorio dentro del frame
    cx, cy = random(0, w), random(0, h)
    
    # 2. Máscara gaussiana suave (hardness controla el borde)
    dist = sqrt((x - cx)² + (y - cy)²) / radius
    mask = clip(1 - dist, 0, 1) ^ hardness
    
    # 3. Aplicar opacidad (0.0 = invisible, 1.0 = opaca)
    mask = mask * opacity
    
    return mask
```

**Parámetros controlables:**
- `radius`: Tamaño de la mancha (en píxeles)
- `opacity`: Intensidad de opacidad (0.7-1.0)
- `hardness`: Suavidad del borde (0.5 = muy suave, 1.0 = abrupto)

#### Paso 3: Aplicación de la Mancha

```python
def apply_stain_to_video(frames, stain_mask):
    for frame in frames:
        # Normalizar frame a [0, 1]
        frame_normalized = frame / 255.0
        
        # Combinar: darkening + tinting
        corrupted = frame_normalized × (1 - mask) + (0.3 × mask)
        
        # Oscurece la zona de la mancha con un tinte
        return clip(corrupted, 0, 1) × 255
```

#### Paso 4: Múltiples Manchas

Para mayor variedad, se agregan **0 a 2 manchas adicionales** por video:
```python
can_manchas = random(0, 2)
for _ in range(can_manchas):
    mask = generate_circular_stain(...)
    video = apply_stain_to_video(video, mask)
```

### Estructura Final del Dataset

```
dataset/
├── batch_0/
│   ├── video_0/
│   │   ├── video_original.mp4      (sin mancha)
│   │   └── video_con_manchas.mp4   (con mancha/s)
│   ├── video_1/
│   │   ├── video_original.mp4
│   │   └── video_con_manchas.mp4
│   └── ...
├── batch_1/
│   └── ... (similar)
└── ...
```

**Estadísticas:**
- Videos por batch: ~60
- Total batches: 20
- Videos totales: ~1200
- Duración: 15 segundos c/u @ 24fps = 360 frames c/u

### División Train/Val

```python
TRAIN_SIZE = 80% de dataset
VAL_SIZE = 20% de dataset

DataLoader:
├── Batch size: 64
├── Shuffle: True (train), False (val)
└── num_workers: 0 (CPU), pin_memory: True (GPU)
```

---

## ⚙️ Funcionalidades

### 1. **Generación del Dataset** (`generacion_dataset.ipynb`)

Automatiza la creación del dataset de entrenamiento:
- Descarga videos MNIST animados
- Corta a duración consistente (15s)
- Genera manchas sintéticas realistas
- Crea pares input/target para supervision

**Entrada:** Videos originales en `./mmnist-medium/`  
**Salida:** Dataset estructura en `./dataset/`

### 2. **Entrenamiento del Modelo** (`trainning_model.ipynb`)

Entrena la U-Net con los datos generados:

```python
# Configuración
CHANNELS_IN = 3
CHANNELS = 64
BATCH_SIZE = 64
LR = 1e-4
EPOCHS = 100

# Loss function
Loss = L1Loss (MAE)
# Más robusto a outliers que MSE

# Optimizer
Optimizer = Adam(lr=1e-4)

# Early Stopping
patience = 7 epochs sin mejora
```

**Monitoreo:**
- Train Loss
- Validation Loss
- Checkpoints cada 10 epochs
- Best model guardado automáticamente

### 3. **Prueba del Modelo** (`test_model.ipynb`)

Demuestra el pipeline completo:

1. **Lectura de video:** `read_video(filename)`
   - Carga todos los frames
   - Convierte BGR → RGB

2. **Síntesis de manchas:** `generate_circular_stain()`
   - Crea manchas realistas
   - Múltiples manchas por video

3. **Aplicación de manchas:** `apply_stain_to_video()`
   - Oscurece zonas
   - Mantiene variabilidad

4. **Procesamiento frame-a-frame:**
   ```python
   for frame in corrupted_video:
       # Normalizar
       frame = RGB2Tensor / 255.0
       
       # Padding a múltiplo de 16 (requerimiento de U-Net)
       frame = pad_to_multiple(frame, 16)
       
       # Inferencia
       with no_grad():
           restored = model(frame)
       
       # Post-procesamiento
       restored = clip(restored × 255, 0, 255)
   ```

5. **Temporal Smoothing:** `temporal_smooth(frames, strength=0.6)`
   - Reduce flickering
   - Mejora coherencia temporal

6. **Guardado:** `save_video(frames, filename, fps=24)`
   - Escribe con codec mp4v
   - Preserva duración original

### 4. **Visualización** (`visualizacion_videos.ipynb`)

Herramientas para inspeccionar resultados:
- Reproductor interactivo de videos
- Comparación original vs corrupto vs restaurado
- Generación de manchas con parámetros ajustables

---

## 📁 Estructura del Proyecto

```
./
├── unet.py                         # Definición del modelo
│   ├── Conv3K                      # Conv 3x3
│   ├── DoubleConv                  # Bloque doble conv
│   ├── DownConv                    # Encoder block
│   ├── UpConv                      # Decoder block
│   ├── UNet                        # Arquitectura completa
│   └── temporal_smooth()           # Post-procesamiento
│
├── generacion_dataset.ipynb        # Crear dataset
├── trainning_model.ipynb           # Entrenar U-Net
├── test_model.ipynb                # Inferencia y validación
├── visualizacion_videos.ipynb      # Herramientas de visualización
│
├── dataset/                        # Dataset generado
│   ├── batch_0/
│   ├── batch_1/
│   └── ... (20 batches)
│
├── mmnist-medium/                  # Videos originales MNIST
│   ├── batch_0_video_0.mp4
│   ├── batch_0_video_1.mp4
│   └── ...
│
├── best_model.pth                  # Mejor modelo entrenado
├── checkpoint_unet_10.pth          # Checkpoints
├── checkpoint_unet_20.pth
└── ...
```

---

## 🚀 Instalación y Uso

### Requisitos

```
Python >= 3.8
PyTorch >= 1.9 (recomendado con CUDA support)
opencv-python >= 4.5
numpy >= 1.19
matplotlib >= 3.3
```

### Instalación

```bash
# 1. Clonar/descargar el proyecto
cd /home/tomy07417/data-science/opcional

# 2. Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

### Uso Paso a Paso

#### Opción 1: Entrenar desde Cero

```bash
# 1. Generar dataset
jupyter notebook generacion_dataset.ipynb
# Ejecutar todas las celdas

# 2. Entrenar modelo
jupyter notebook trainning_model.ipynb
# Configurar parámetros si es necesario
# Ejecutar entrenamiento (puede tomar varias horas)

# Resultado: best_model.pth
```

#### Opción 2: Usar Modelo Pre-entrenado

```bash
# 1. Usar test_model.ipynb directamente
jupyter notebook test_model.ipynb

# 2. Modificar video de entrada:
# Línea 2: frames = read_video("./tu_video.mp4")

# 3. Ejecutar celdas en orden
# Resultado: video_reconstruido.mp4
```

#### Opción 3: Script Personalizado

```python
import torch
import cv2
import numpy as np
from unet import UNet, temporal_smooth

# Cargar modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(3, 64).to(device)
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

# Procesar video
cap = cv2.VideoCapture("video_input.mp4")
frames_out = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Normalizar y convertir
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # Padding
    _, H, W = frame_tensor.shape[1:]
    pad_h = (16 - H % 16) % 16
    pad_w = (16 - W % 16) % 16
    if pad_h > 0 or pad_w > 0:
        frame_tensor = torch.nn.functional.pad(frame_tensor, (0, pad_w, 0, pad_h))
    
    # Inferencia
    with torch.no_grad():
        output = model(frame_tensor)
    
    # Desnormalizar
    output_frame = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    output_frame = np.clip(output_frame * 255, 0, 255).astype(np.uint8)
    
    frames_out.append(output_frame)

cap.release()

# Suavizado temporal y guardado
frames_smooth = temporal_smooth(np.array(frames_out), strength=0.6)
# ... guardar video
```

---

## 📈 Resultados

### Métricas de Entrenamiento

Monitoreadas durante el entrenamiento:
- **Train Loss:** Disminuye constantemente (modelo aprendiendo)
- **Validation Loss:** Valida generalización
- **Early Stopping:** Si val_loss no mejora en 7 epochs

### Ejemplos Visuales

**Video Original (limpio)**
- MNIST dígito moviéndose
- Fondo uniforme

**Video Corrupto (con mancha)**
- Mancha circular oscura fija
- Dígito se mueve detrás

**Video Restaurado (salida modelo)**
- Mancha removida por U-Net
- Temporal smoothing elimina flickering
- Reconstrucción clara del movimiento

---

## 🔮 Mejoras Futuras

### Corto Plazo

1. **Arquitecturas Alternativas**
   - ResNet para baseline
   - DenseNet para mejor flujo de características
   - Attention mechanisms para enfoque en manchas

2. **Variabilidad de Manchas**
   - Manchas irregulares (no solo círculos)
   - Manchas que varían en intensidad
   - Múltiples manchas con diferentes opacidades

3. **Optimización**
   - Reducir tamaño del modelo
   - Quantización para inferencia más rápida
   - TorchScript para deployment

### Mediano Plazo

4. **Modelos Temporales**
   - ConvLSTM para procesar secuencias
   - 3D-UNet que vea múltiples frames simultáneamente
   - Transformers para relaciones de largo plazo

5. **Datos Reales**
   - Recolectar videos reales con manchas
   - Fine-tuning con datos sintéticos + reales
   - Dataset de diferentes tipos de defectos

6. **Video Más Complejo**
   - Escenas naturales (no solo MNIST)
   - Múltiples objetos
   - Oclusiones y movimientos rápidos

### Largo Plazo

7. **Arquitecturas de Estado del Arte**
   - Blind inpainting con GANs
   - Diffusion models para reconstrucción
   - Multi-scale processing

8. **Aplicaciones Real-World**
   - Streaming en vivo
   - Cámaras de vigilancia
   - Procesamiento de video histórico
   - Restauración de filmografía antigua

---

## 📚 Referencias

- **U-Net Paper:** [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- **Dataset:** [Moving MNIST Dataset - Kaggle](https://www.kaggle.com/datasets/yichengs/captioned-moving-mnist-dataset-medium-version)
- **PyTorch Docs:** [pytorch.org](https://pytorch.org/)

---

## 👨‍💼 Autor

**Tomás Amundarain**  
TP N°4 - Eliminación de Manchas en Videos  
Diciembre 2025

---

## 📝 Licencia

Este proyecto es de código abierto para propósitos educativos.
