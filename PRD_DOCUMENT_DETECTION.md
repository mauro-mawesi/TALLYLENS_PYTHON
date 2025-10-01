# PRD: Document Detection & Perspective Correction System

**Versión:** 3.0
**Fecha:** 2025-09-30
**Estado:** Diseño Final antes de Implementación

---

## 1. PROBLEMA A RESOLVER

### 1.1 Situación Actual
- ✅ DocTR rota correctamente las imágenes (horizontal → vertical)
- ❌ DocTR NO elimina el fondo (solo hace bounding box del texto)
- ❌ Las imágenes procesadas mantienen mucho fondo de madera
- ❌ El OCR podría confundirse con las vetas de la madera

### 1.2 Objetivo
Crear un sistema de detección de documentos que:
1. Detecte los **4 bordes físicos del recibo** (no solo el texto)
2. Aplique **corrección de perspectiva** si el recibo está en ángulo
3. Haga **crop preciso** removiendo TODO el fondo
4. Preserve la **calidad** del documento
5. Sea **robusto** (funcione en 90%+ de casos)

---

## 2. ARQUITECTURA DEL SISTEMA

### 2.1 Pipeline Completo

```
┌─────────────────────────────────────────────────────────────┐
│                    IMAGEN DE ENTRADA                        │
│          (Recibo sobre mesa de madera)                      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────┐
│              PASO 1: PREPROCESAMIENTO                       │
│                                                             │
│  • Resize a max 2000px (performance)                        │
│  • Convertir BGR → Grayscale                                │
│  • Aplicar Gaussian Blur (5x5)                              │
│  • Calcular aspect ratio (detectar orientación)             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────┐
│         PASO 2: DETECCIÓN DE BORDES DEL DOCUMENTO          │
│                                                             │
│  Estrategia Multi-Método (probar en orden):                 │
│                                                             │
│  Método 1: ADAPTIVE THRESHOLD + CONTOURS                    │
│    • Adaptive threshold (GAUSSIAN, blockSize=11)            │
│    • Morfología: dilate + erode (cerrar gaps)               │
│    • Find contours (RETR_EXTERNAL)                          │
│    • Buscar quadrilateral más grande (approxPolyDP)         │
│    ✓ Funciona: Fondos uniformes, buena iluminación         │
│    ✗ Falla: Fondos complejos (madera con vetas)            │
│                                                             │
│  Método 2: CANNY EDGE DETECTION + HOUGH LINES               │
│    • Canny edges (50, 150)                                  │
│    • Detectar líneas con HoughLinesP                        │
│    • Agrupar líneas por orientación (H/V)                   │
│    • Encontrar intersecciones → 4 esquinas                  │
│    ✓ Funciona: Fondos complejos, documento con bordes      │
│    ✗ Falla: Recibos arrugados, bordes poco definidos       │
│                                                             │
│  Método 3: DOCTR TEXT DETECTION (Fallback ML)               │
│    • Usar DocTR para detectar regiones de texto             │
│    • Calcular convex hull de TODAS las detecciones          │
│    • Añadir margen 10–15% alrededor del hull                │
│    ✓ Funciona: Siempre (si hay texto)                      │
│    ✗ Falla: No detecta bordes reales, solo texto           │
│                                                             │
│  Selección: Usar resultado con mejor score (ver 3.3)       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────┐
│        PASO 3: VALIDACIÓN Y SCORING DE DETECCIÓN           │
│                                                             │
│  Para cada método que detectó 4 esquinas:                   │
│                                                             │
│  Validaciones:                                              │
│    1. ¿Tiene exactamente 4 esquinas? [Requerido]            │
│    2. ¿Área > 20% de imagen? [Requerido]                    │
│    3. ¿Forma un cuadrilátero convexo? [Recomendado]         │
│    4. ¿Ángulos entre 60° y 120°? [Recomendado]              │
│    5. ¿Lados con ratio razonable? [Opcional]                │
│                                                             │
│  Scoring (0-100):                                           │
│    • +40 puntos: Área cubre 35-85% de imagen (ideal)        │
│    • +20 puntos: Todos los ángulos cerca de 90°             │
│    • +20 puntos: Lados similares + paralelismo de opuestos  │
│    • +10 puntos: Cuadrilátero es convexo                    │
│    • +10 puntos: Esquinas no están en bordes de imagen      │
│                                                             │
│  Decisión:                                                  │
│    • Score >= 70: APLICAR corrección de perspectiva         │
│    • Score 40-69: APLICAR solo crop (no warp)               │
│    • Score < 40: USAR imagen original (no procesar)         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────┐
│      PASO 4: CORRECCIÓN DE PERSPECTIVA (si score >= 70)    │
│                                                             │
│  1. Ordenar esquinas: TL, TR, BR, BL                        │
│     • TL = min(x + y)                                       │
│     • BR = max(x + y)                                       │
│     • TR = min(y - x)                                       │
│     • BL = max(y - x)                                       │
│                                                             │
│  2. Calcular dimensiones del documento rectificado:         │
│     • width = max(dist(TL→TR), dist(BL→BR))                 │
│     • height = max(dist(TL→BL), dist(TR→BR))                │
│                                                             │
│  3. Aplicar transformación de perspectiva:                  │
│     • M = getPerspectiveTransform(src, dst)                 │
│     • warped = warpPerspective(img, M, (width, height),     │
│       flags=INTER_CUBIC si amplía, INTER_AREA si reduce)    │
│                                                             │
│  4. Resultado: Documento "plano" visto desde arriba         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────┐
│            PASO 5: CROP FINAL Y LIMPIEZA                    │
│                                                             │
│  1. Si se aplicó warp: imagen ya está cropeada              │
│  2. Si solo crop: usar las 4 esquinas como bounding box     │
│  3. Agregar margen pequeño (2-3%)                           │
│  4. Asegurar que crop esté dentro de límites de imagen      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────┐
│          PASO 6: DETECCIÓN Y CORRECCIÓN DE ORIENTACIÓN     │
│                                                             │
│  1. Detectar orientación de texto (OSD)                     │
│     • Preferir DocTR/Tesseract OSD si disponible            │
│     • Registrar método usado en metadata                    │
│                                                             │
│  2. Fallback por aspect ratio                               │
│     • ratio = width / height; si ratio > 1.3 → rotar 90°    │
│     • Solo si no hay señal confiable de texto               │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────┐
│              PASO 7: POST-PROCESAMIENTO FINAL               │
│                                                             │
│  1. Resize si excede límites (max 4000x4000)                │
│  2. Aplicar filtros sutiles (OPCIONAL, desactivado default):│
│     • Sharpen ligero (solo si imagen está borrosa)          │
│     • Denoise suave (bilateral filter)                      │
│  3. Codificar como PNG lossless (quality 100)               │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────┐
│                  IMAGEN FINAL PROCESADA                     │
│         (Recibo vertical, sin fondo, buena calidad)         │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. ESPECIFICACIÓN TÉCNICA DETALLADA

### 3.1 Método 1: Adaptive Threshold + Contours

**Pseudocódigo:**
```python
def detect_document_adaptive(gray_img):
    # 1. Threshold adaptativo
    thresh = cv2.adaptiveThreshold(
        gray_img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2
    )

    # 2. Morfología para cerrar gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # 3. Detectar contornos
    contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 4. Ordenar por área (más grande primero)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # 5. Buscar quadrilateral en top 5 contours
    for contour in contours[:5]:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) == 4:
            area = cv2.contourArea(approx)
            img_area = gray_img.shape[0] * gray_img.shape[1]

            if area > img_area * 0.20:  # Al menos 20% del área
                return approx.reshape(4, 2), calculate_score(approx.reshape(4, 2), gray_img.shape)

    return None, 0
```

**Parámetros críticos:**
- `blockSize=11`: Tamaño de ventana para threshold adaptativo
- `C=2`: Constante de ajuste
- `0.02 * peri`: Tolerancia para aproximación de polígono
- `0.20`: Mínimo área del documento (20% de imagen)

**Casos de éxito:**
- ✅ Recibo blanco sobre fondo oscuro uniforme
- ✅ Buena iluminación, sin sombras fuertes
- ✅ Documento plano (no arrugado)

**Casos de fallo:**
- ❌ Fondo complejo (madera con vetas)
- ❌ Iluminación no uniforme
- ❌ Documento del mismo color que el fondo

---

### 3.2 Método 2: Canny + Hough Lines (clustering de ángulos)

**Pseudocódigo:**
```python
def detect_document_hough(gray_img):
    # 1. Edge detection
    edges = cv2.Canny(gray_img, 50, 150, apertureSize=3)

    # 2. Detectar líneas
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=100,
        minLineLength=100,
        maxLineGap=10
    )

    if lines is None:
        return None, 0

    # 3. Clasificar líneas en horizontales y verticales
    h_lines = []  # Líneas horizontales (ángulo ~0° o 180°)
    v_lines = []  # Líneas verticales (ángulo ~90°)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

        if abs(angle) < 10 or abs(angle - 180) < 10:
            h_lines.append(line)
        elif abs(angle - 90) < 10 or abs(angle + 90) < 10:
            v_lines.append(line)

    # 4. Encontrar líneas extremas (bordes del documento)
    if len(h_lines) < 2 or len(v_lines) < 2:
        return None, 0

    top_line = min(h_lines, key=lambda l: l[0][1])
    bottom_line = max(h_lines, key=lambda l: l[0][1])
    left_line = min(v_lines, key=lambda l: l[0][0])
    right_line = max(v_lines, key=lambda l: l[0][0])

    # 5. Calcular intersecciones para obtener 4 esquinas
    corners = [
        line_intersection(top_line, left_line),    # TL
        line_intersection(top_line, right_line),   # TR
        line_intersection(bottom_line, right_line),# BR
        line_intersection(bottom_line, left_line)  # BL
    ]

    return np.array(corners), calculate_score(np.array(corners), gray_img.shape)
```

**Parámetros/consideraciones:**
- `threshold=100`: Mínimo número de intersecciones para detectar línea
- `minLineLength=100`: Longitud mínima de línea (px)
- `maxLineGap=10`: Máximo gap entre segmentos de línea
- Clasificar líneas por clusters de ángulos (k=2) y tomar extremos
- Alternativa: usar LSD (`cv2.ximgproc.createFastLineDetector`) y fusión de segmentos

**Casos de éxito:**
- ✅ Documentos con bordes bien definidos
- ✅ Fondos complejos (madera, tela)
- ✅ Buena iluminación

**Casos de fallo:**
- ❌ Bordes del documento difusos o arrugados
- ❌ Iluminación muy baja
- ❌ Documentos transparentes

---

### 3.3 Scoring System

**Función de scoring:**
```python
def calculate_score(corners, img_shape):
    score = 0
    h, w = img_shape[:2]
    img_area = h * w

    # 1. Score por área (40 puntos)
    quad_area = cv2.contourArea(corners)
    area_ratio = quad_area / img_area

    if 0.35 <= area_ratio <= 0.85:
        score += 40  # Área ideal
    elif 0.20 <= area_ratio < 0.35:
        score += 30  # Área pequeña pero aceptable
    elif 0.85 < area_ratio <= 0.95:
        score += 25  # Área grande (puede incluir algo de fondo)
    else:
        return 0  # Área inválida

    # 2. Score por ángulos (20 puntos)
    angles = calculate_corner_angles(corners)
    angle_score = 0
    for angle in angles:
        if 80 <= angle <= 100:
            angle_score += 5  # Ángulo muy bueno (~90°)
        elif 70 <= angle <= 110:
            angle_score += 3  # Ángulo aceptable
    score += angle_score

    # 3. Score por simetría de lados y paralelismo (20 puntos)
    sides = [
        np.linalg.norm(corners[1] - corners[0]),  # Top
        np.linalg.norm(corners[2] - corners[1]),  # Right
        np.linalg.norm(corners[3] - corners[2]),  # Bottom
        np.linalg.norm(corners[0] - corners[3])   # Left
    ]

    top_bottom_ratio = min(sides[0], sides[2]) / max(sides[0], sides[2])
    left_right_ratio = min(sides[1], sides[3]) / max(sides[1], sides[3])

    parallel_bonus = parallelism_score(corners)  # 0..5
    if top_bottom_ratio > 0.90 and left_right_ratio > 0.90:
        score += 15 + parallel_bonus
    elif top_bottom_ratio > 0.80 and left_right_ratio > 0.80:
        score += 10 + parallel_bonus

    # 4. Score por convexidad (10 puntos)
    if cv2.isContourConvex(corners):
        score += 10

    # 5. Score por distancia a bordes (10 puntos)
    # Las esquinas NO deben estar en los bordes de la imagen
    margin = 0.05  # 5% margin
    corners_on_edge = 0
    for corner in corners:
        x, y = corner
        if (x < w * margin or x > w * (1 - margin) or
            y < h * margin or y > h * (1 - margin)):
            corners_on_edge += 1

    if corners_on_edge == 0:
        score += 10
    elif corners_on_edge <= 2:
        score += 5

    return score
```

**Interpretación de scores:**
- **90-100**: Detección perfecta → Aplicar warp con confianza total
- **70-89**: Detección buena → Aplicar warp
- **40-69**: Detección dudosa → Solo crop (no warp)
- **0-39**: Detección fallida → Usar imagen original

---

## 4. INTEGRACIÓN CON BACKEND

### 4.1 Request del Backend → Python

**Endpoint:** `POST /process-receipt`

**Payload (recomendado):**
```json
{
  "fileId": "1759063603011-174833084.jpg",
  "relativePath": "uploads/1759063603011-174833084.jpg"
}
```

Notas:
- Evitar rutas absolutas por seguridad. Resolver `relativePath` dentro de un root seguro o mapear `fileId`.

**Headers:**
```
Content-Type: application/json
```

---

### 4.2 Response Python → Backend

**Success (200 OK):**
```json
{
  "success": true,
  "processed": true,
  "processedPath": "/home/mauricio/projects/LAB/RECIBOS_APP/backend/uploads/1759063603011-174833084_processed.png",
  "format": "png",
  "metadata": {
    "originalDimensions": {
      "width": 4000,
      "height": 1848
    },
    "processedDimensions": {
      "width": 3200,
      "height": 900
    },
    "detectionMethod": "adaptive_threshold",
    "detectionScore": 85,
    "perspectiveCorrectionApplied": true,
    "rotationApplied": true,
    "rotationAngle": 90,
    "keptAreaPercent": 39.0,
    "orientationMethod": "text_osd",
    "colorSpace": "RGB",
    "processingTimeMs": 1250
  }
}
```

**Failure - Detection Failed (200 OK - imagen sin procesar):**
```json
{
  "success": false,
  "processed": false,
  "processedPath": "/home/mauricio/projects/LAB/RECIBOS_APP/backend/uploads/1759063603011-174833084.jpg",
  "format": "png",
  "metadata": {
    "originalDimensions": {
      "width": 4000,
      "height": 1848
    },
    "processedDimensions": {
      "width": 4000,
      "height": 1848
    },
    "detectionMethod": "none",
    "detectionScore": 0,
    "perspectiveCorrectionApplied": false,
    "rotationApplied": false,
    "keptAreaPercent": 100.0,
    "orientationMethod": "aspect_ratio",
    "colorSpace": "RGB",
    "processingTimeMs": 450,
    "warning": "Document detection failed, using original image"
  }
}
```

**Error (500 Internal Server Error):**
```json
{
  "error": "Failed to process image",
  "details": "File not found: /path/to/image.jpg"
}
```

---

### 4.3 Metadata Explicada

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `detectionMethod` | string | Método usado: `"adaptive_threshold"`, `"canny_hough"`, `"doctr_fallback"`, `"none"` |
| `detectionScore` | int | Score 0-100 de confianza en la detección |
| `perspectiveCorrectionApplied` | bool | Si se aplicó warp de perspectiva |
| `rotationApplied` | bool | Si se rotó la imagen |
| `rotationAngle` | int | Ángulo de rotación: 0, 90, 180, 270 |
| `keptAreaPercent` | float | % de área conservada respecto al original (100 = sin recorte) |
| `orientationMethod` | string | Método de orientación (`"text_osd"`, `"aspect_ratio"`, `"none"`) |
| `colorSpace` | string | Espacio de color de salida (`"RGB"` o `"GRAY"`) |
| `processingTimeMs` | int | Tiempo de procesamiento en milisegundos |

---

## 5. CASOS DE USO Y RESULTADOS ESPERADOS

### 5.1 Caso Ideal: Recibo sobre Mesa Oscura

**Input:**
- Recibo blanco sobre mesa de madera oscura
- Buena iluminación
- Recibo horizontal
- Fondo visible alrededor

**Procesamiento:**
1. Método 1 (Adaptive) detecta documento con score 92
2. Aplica corrección de perspectiva
3. Rota 90° a vertical
4. Crop preciso

**Output:**
- ✅ Recibo vertical
- ✅ 0% de fondo visible
- ✅ Bordes limpios
- ✅ Calidad preservada

**Metadata esperada:**
```json
{
  "detectionMethod": "adaptive_threshold",
  "detectionScore": 92,
  "perspectiveCorrectionApplied": true,
  "rotationApplied": true,
  "keptAreaPercent": 39.0
}
```

---

### 5.2 Caso Complejo: Recibo sobre Madera con Vetas

**Input:**
- Recibo sobre mesa de madera clara con vetas
- Contraste bajo entre documento y fondo
- Recibo ligeramente en ángulo

**Procesamiento:**
1. Método 1 (Adaptive) falla: score 25
2. Método 2 (Canny+Hough) detecta con score 72
3. Aplica corrección de perspectiva
4. Rota si es necesario
5. Crop

**Output:**
- ✅ Recibo rectificado
- ✅ ~5% de fondo visible (aceptable)
- ✅ Bordes limpios
- ✅ Calidad preservada

**Metadata esperada:**
```json
{
  "detectionMethod": "canny_hough",
  "detectionScore": 72,
  "perspectiveCorrectionApplied": true,
  "rotationApplied": false,
  "keptAreaPercent": 45.0
}
```

---

### 5.3 Caso Difícil: Recibo Arrugado

**Input:**
- Recibo arrugado o doblado
- Bordes no definidos
- Fondo complejo

**Procesamiento:**
1. Método 1 falla: score 15
2. Método 2 falla: score 30
3. Método 3 (DocTR fallback) detecta texto: score 55
4. Aplica solo crop (NO warp) por score bajo
5. Rota si es necesario

**Output:**
- ✅ Recibo orientado correctamente
- ⚠️ ~15% de fondo visible
- ✅ Sin distorsión por warp incorrecto
- ✅ Calidad preservada

**Metadata esperada:**
```json
{
  "detectionMethod": "doctr_fallback",
  "detectionScore": 55,
  "perspectiveCorrectionApplied": false,
  "rotationApplied": true,
  "keptAreaPercent": 60.0,
  "warning": "Low confidence detection, perspective correction skipped"
}
```

---

### 5.4 Caso Extremo: Foto Muy Mala

**Input:**
- Imagen borrosa
- Muy poca luz
- Documento apenas visible

**Procesamiento:**
1. Todos los métodos fallan: score < 40
2. Sistema devuelve imagen original sin procesar
3. Solo aplica orientación básica (aspect ratio)

**Output:**
- ⚠️ Imagen original (con posible rotación)
- ❌ No se eliminó fondo
- ✅ No se distorsionó la imagen

**Metadata esperada:**
```json
{
  "detectionMethod": "none",
  "detectionScore": 0,
  "perspectiveCorrectionApplied": false,
  "rotationApplied": true,
  "keptAreaPercent": 100.0,
  "warning": "Document detection failed, using original image"
}
```

---

## 6. TESTING Y VALIDACIÓN

### 6.1 Test Cases

| # | Descripción | Input | Expected Output | Priority |
|---|-------------|-------|-----------------|----------|
| 1 | Recibo horizontal sobre mesa oscura | `tests/1759070441380.jpg` | Vertical, sin fondo, IoU quad ≥ 0.85, score > 80 | P0 |
| 2 | Recibo horizontal sobre madera clara | `tests/1759063603011.jpg` | Vertical, fondo residual < 10%, score > 70 | P0 |
| 3 | Recibo ya vertical y cropeado | `tests/1759154598361.jpg` | Sin cambios (keptAreaPercent ~100), score > 60 | P1 |
| 4 | Recibo en ángulo 30° | Mock image | Rectificado, ángulo residual < 2°, score > 75 | P0 |
| 5 | Recibo muy pequeño en imagen grande | Mock image | Cropeado, IoU quad ≥ 0.8 | P1 |
| 6 | Múltiples recibos en una foto | Mock image | Detecta el más grande correctamente | P2 |
| 7 | Foto borrosa | Mock image | Original sin procesar (processed=false) | P1 |

### 6.2 Success Metrics

**Objetivo Mínimo (MVP):**
- ✅ 70% de imágenes con score > 70
- ✅ IoU quad mediano ≥ 0.8 en casos exitosos
- ✅ 0% de warps incorrectos (validado por checks geométricos)
- ✅ Tiempo de procesamiento < 3s por imagen

**Objetivo Ideal:**
- ✅ 85% de imágenes con score > 70
- ✅ IoU quad mediano ≥ 0.85; fondo residual < 5%
- ✅ Reducción del CER/WER del OCR ≥ 10% vs baseline
- ✅ Tiempo de procesamiento < 2s por imagen

### 6.3 Logging y Métricas

**Nivel INFO:**
```
INFO: Processing image: (4000, 1848, 3)
INFO: Method 1 (adaptive): score=32 (area=10, angles=12, symmetry=5, convex=5), rejected
INFO: Method 2 (canny_hough): score=78 (area=30, angles=20, symmetry=18, parallel=2, edges=8), selected
INFO: Applying perspective correction
INFO: Rotating image 90° clockwise
INFO: Final dimensions: (3200, 900)
INFO: Processing completed in 1.2s
```

**Nivel DEBUG:**
```
DEBUG: Adaptive threshold: found 3 contours
DEBUG: Top contour area: 5200000 (78% of image)
DEBUG: Approximated to 5 points (need 4), rejected
DEBUG: Canny edges: 12500 edge pixels
DEBUG: Hough lines: 48 horizontal, 52 vertical
DEBUG: Corners detected: [(120, 80), (3890, 95), (3875, 1760), (105, 1745)]
DEBUG: Corner angles: [88.3°, 91.2°, 89.8°, 90.7°]
DEBUG: Score breakdown: area=38, angles=20, symmetry=18, parallel=2, convex=10, edges=10 → Total=88

Métricas operativas:
- Contadores por método elegido, tasas de processed=true/false
- Distribución de scores y tiempos por paso
- Tasa de rotación aplicada y método de orientación usado
```

---

## 7. IMPLEMENTACIÓN

### 7.1 Archivos a Crear/Modificar

**Nuevos archivos:**
1. `app/document_processor_hybrid.py` - Nueva implementación con multi-método
2. `app/utils/geometry.py` - Funciones de geometría (intersecciones, ángulos, ordering TL-TR-BR-BL)
3. `app/utils/scoring.py` - Sistema de scoring (incluye parallelism_score)
4. `app/utils/orientation.py` - Detección de orientación de texto (DocTR/OSD) y fallback por aspect ratio
5. `tests/test_document_detection.py` - Tests unitarios y de regresión

**Archivos a modificar:**
1. `app/server.py` - Cambiar a usar nuevo procesador
2. `requirements.txt` - Agregar `imutils`, `scikit-image`

### 7.2 Dependencias Adicionales

```txt
opencv-python==4.9.*
opencv-contrib-python==4.9.*   # opcional para LSD y ximgproc
imutils==0.5.4
scikit-image==0.22.0
python-doctr==0.6.*            # fallback ML y orientación por texto
# pytesseract==0.3.*          # opcional si se usa OSD de Tesseract
```

### 7.3 Configuración

```python
# config.py
class ProcessingConfig:
    # Preprocesamiento
    MAX_DIMENSION_DETECTION = 2000  # Max size para detección (performance)
    MAX_DIMENSION_OUTPUT = 4000     # Max size output
    USE_CLAHE = False               # Mejora de contraste (opcional)
    USE_SHADOW_REMOVAL = False      # Remover sombras suaves (opcional)

    # Detección
    MIN_DOCUMENT_AREA_RATIO = 0.20  # Mínimo 20% del área
    MAX_DOCUMENT_AREA_RATIO = 0.95  # Máximo 95% del área
    METHOD_ORDER = ["adaptive_threshold", "canny_hough", "doctr_fallback"]

    # Scoring
    MIN_SCORE_FOR_WARP = 70         # Score mínimo para aplicar warp
    MIN_SCORE_FOR_CROP = 40         # Score mínimo para aplicar crop

    # Adaptive Threshold
    ADAPTIVE_BLOCK_SIZE = 11
    ADAPTIVE_C = 2

    # Canny
    CANNY_THRESHOLD1 = 50
    CANNY_THRESHOLD2 = 150

    # Hough Lines
    HOUGH_THRESHOLD = 100
    HOUGH_MIN_LINE_LENGTH = 100
    HOUGH_MAX_LINE_GAP = 10

    # Orientación
    USE_TEXT_ORIENTATION = True     # Preferir orientación por texto (DocTR/OSD)
    HORIZONTAL_ASPECT_RATIO = 1.3   # Fallback: si ratio > 1.3, está horizontal

    # Output
    OUTPUT_FORMAT = 'png'           # 'png' o 'webp_lossless'
    OUTPUT_QUALITY = 100
    OUTPUT_COLORSPACE = 'RGB'       # 'RGB' o 'GRAY'
    SAVE_DEBUG_IMAGES = False
    DEBUG_DIR = 'debug/'
    DOCTR_MARGIN_PERCENT = 0.12     # Margen alrededor del hull de texto
```

---

## 8. ROLLOUT PLAN

### Fase 1: Implementación (2-3 horas)
1. ✅ Crear estructura de archivos
2. ✅ Implementar método 1 (Adaptive)
3. ✅ Implementar método 2 (Canny+Hough)
4. ✅ Implementar sistema de scoring
5. ✅ Integrar con servidor Flask

### Fase 2: Testing Inicial (30 min)
1. ✅ Probar con 3 imágenes de test
2. ✅ Validar que scores sean razonables
3. ✅ Ajustar parámetros si es necesario

### Fase 3: Testing Completo (1 hora)
1. ✅ Procesar todas las imágenes de `tests/`
2. ✅ Comparar resultados con versión anterior
3. ✅ Medir success rate

### Fase 4: Ajuste Fino (1 hora)
1. ✅ Ajustar parámetros basándose en resultados
2. ✅ Agregar casos especiales si es necesario
3. ✅ Re-test

### Fase 5: Deploy (15 min)
1. ✅ Actualizar documentación
2. ✅ Commit y push
3. ✅ Reiniciar servicio

---

## 9. RIESGOS Y MITIGACIONES

| Riesgo | Probabilidad | Impacto | Mitigación |
|--------|--------------|---------|------------|
| Detección falla en 30%+ casos | Alta | Alto | Usar DocTR como fallback obligatorio |
| Warp distorsiona documentos | Media | Alto | Sistema de scoring conservador (threshold 70) |
| Performance lento (>5s) | Baja | Medio | Resize a 2000px antes de detección |
| Memoria insuficiente | Baja | Alto | Limitar workers a 1, timeout alto |
| OpenCV falla en algunas imágenes | Baja | Medio | Try-catch en cada método, fallback |
| Orientación errónea | Media | Medio | Preferir orientación por texto y registrar método |

---

## 10. DECISIONES DE DISEÑO

### 10.1 ¿Por qué Multi-Método?
- **Problema:** Ningún método funciona en todos los casos
- **Solución:** Probar múltiples métodos y elegir el mejor
- **Ventaja:** Mayor tasa de éxito (85-90% vs 60-70%)

### 10.2 ¿Por qué Sistema de Scoring?
- **Problema:** Detecciones incorrectas distorsionan la imagen
- **Solución:** Solo aplicar warp si hay alta confianza
- **Ventaja:** Evita degradar imágenes que ya están bien

### 10.3 ¿Por qué NO usar solo ML?
- **Problema:** DocTR detecta texto, no bordes físicos
- **Solución:** Combinar ML (DocTR) con CV tradicional (OpenCV)
- **Ventaja:** Lo mejor de ambos mundos

### 10.4 ¿Por qué PNG Lossless?
- **Problema:** JPEG degrada calidad en cada procesamiento
- **Solución:** Usar PNG para preservar 100% calidad
- **Trade-off:** Archivos más grandes (~2-3x) pero calidad perfecta para OCR
- Alternativa configurable: `webp_lossless` o salida en escala de grises para reducir tamaño

---

## 11. PREGUNTAS PARA VALIDAR ANTES DE IMPLEMENTAR

1. ✅ ¿El backend puede manejar archivos PNG de ~5-10MB?
2. ✅ ¿El timeout de 300s es suficiente para la primera carga del modelo?
3. ✅ ¿Está bien que el servicio Python use ~500MB de RAM?
4. ⚠️ ¿Aprobamos cambiar el contrato a `POST /process-receipt` con `fileId/relativePath` (sin rutas absolutas)?
5. ⚠️ ¿Quieres que se guarde un log de metadata en archivo para debugging?
6. ⚠️ ¿Quieres que se guarden imágenes intermedias (debug) en desarrollo? ¿Cuánto espacio reservar?
7. ⚠️ ¿Preferencias sobre `OUTPUT_COLORSPACE` (RGB vs GRAY) y `webp_lossless`?

---

## 12. APROBACIÓN

**Este PRD debe ser aprobado antes de implementar.**

**Cambios propuestos respecto a versión actual:**
1. ✅ Reemplazar DocTR como detector principal por OpenCV multi-método
2. ✅ Mantener DocTR solo como fallback para casos difíciles
3. ✅ Agregar sistema de scoring robusto
4. ✅ Agregar metadata detallada en response
5. ✅ Configuración más granular

**Aprobado por:** [Pendiente]
**Fecha:** [Pendiente]

---

**FIN DEL PRD**
