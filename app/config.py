class ProcessingConfig:
    # Preprocesamiento
    MAX_DIMENSION_DETECTION = 1280  # Modo low-resource: menor tamaño para detección
    MAX_DIMENSION_OUTPUT = 4000     # Max size output
    USE_CLAHE = False               # Mejora de contraste (opcional)
    USE_SHADOW_REMOVAL = False      # Remover sombras suaves (opcional)

    # Detección
    MIN_DOCUMENT_AREA_RATIO = 0.20  # Mínimo 20% del área
    MAX_DOCUMENT_AREA_RATIO = 0.95  # Máximo 95% del área
    # Modo low-resource: priorizar color, luego GrabCut centrado, líneas y contornos
    METHOD_ORDER = ["paper_color", "grabcut_center", "lsd_lines", "adaptive_threshold", "doctr_fallback"]

    # Scoring
    MIN_SCORE_FOR_WARP = 80         # Más estricto para evitar warps malos
    MIN_SCORE_FOR_CROP = 40         # Score mínimo para aplicar crop
    RECTANGULARITY_MIN_FOR_WARP = 0.88
    QUAD_AREA_MIN_RATIO = 0.20      # Área del quad vs imagen para warp
    QUAD_AREA_MAX_RATIO = 0.80
    DOC_ASPECT_MIN = 1.15          # Min ratio max(w,h)/min(w,h) plausible for recibos
    DOC_ASPECT_MAX = 7.0           # Max ratio plausible

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
    OUTPUT_FORMAT = 'webp'          # Preferir WebP lossy para tamaño razonable
    OUTPUT_QUALITY = 90
    OUTPUT_COLORSPACE = 'RGB'       # 'RGB' o 'GRAY'
    SAVE_DEBUG_IMAGES = False
    DEBUG_DIR = 'debug/'
    DOCTR_MARGIN_PERCENT = 0.12     # Margen alrededor del hull de texto

    # ML Detector (ONNX)
    USE_ML_DETECTOR = False
    ML_MODEL_PATH = os.getenv('ML_MODEL_PATH', 'models/document_boundary.onnx') if 'os' in globals() else 'models/document_boundary.onnx'
    ML_CONF_THRESHOLD = 0.25
    ML_IOU_THRESHOLD = 0.50

    # Salida
    ALWAYS_WRITE_PROCESSED = True    # Forzar creación de archivo _processed aun sin cambios
    ENABLE_GRABCUT_REFINE = True     # Refinar bordes con GrabCut antes de warpear
    # Seguridad de recortes y trimming
    SAFE_MARGIN_PERCENT = 0.02       # Margen extra alrededor del crop (2%)
    ENABLE_TRIM = True               # Permitir recorte de bordes uniformes post-proceso
    TRIM_TOL = 6                     # Tolerancia de uniformidad
    TRIM_MAX_FRAC_X = 0.08           # Máximo 8% por lado en X
    TRIM_MAX_FRAC_Y = 0.02           # Máximo 2% por lado en Y (evitar cortar encabezados/pies)

    # Perfil de mejora tipo MLKit
    ENHANCE_PROFILE = 'mlkit'        # 'mlkit' | 'none'
    ENHANCE_MODE = 'color'           # 'color' | 'bw' | 'none'
    PAD_BORDER_PERCENT = 0.03        # Margen blanco alrededor del recibo
    CLAHE_CLIP = 3.0
    CLAHE_TILE = 8

    # OCR (fallback local)
    OCR_ENABLED = True
    OCR_LANG = os.getenv('OCR_LANG', 'eng') if 'os' in globals() else 'eng'
    OCR_OEM = int(os.getenv('OCR_OEM', '3')) if 'os' in globals() else 3  # 0..3
    OCR_PSM = int(os.getenv('OCR_PSM', '6')) if 'os' in globals() else 6  # 3/6 buenos generales
    OCR_MIN_CONF = int(os.getenv('OCR_MIN_CONF', '40')) if 'os' in globals() else 40
