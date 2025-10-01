from typing import Any, Dict, List, Optional, Tuple
import os
import logging

import cv2
import numpy as np

try:
    import pytesseract  # type: ignore
    HAS_TESSERACT = True
except Exception:
    HAS_TESSERACT = False

from .config import ProcessingConfig as C
from .document_processor_hybrid import _enhance_mlkit_style


def _prefer_processed_path(input_path: str) -> Optional[str]:
    base, _ = os.path.splitext(input_path)
    for ext in ('.png', '.webp', '.jpg', '.jpeg'):
        cand = f"{base}_processed{ext}"
        if os.path.isfile(cand):
            return cand
    return None


def _binarize(gray: np.ndarray) -> np.ndarray:
    # Umbral adaptativo tipo MLKit bw
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10)


def _prepare_for_ocr(bgr: np.ndarray) -> np.ndarray:
    # Aplicar perfil MLKit si corresponde
    if C.ENHANCE_PROFILE == 'mlkit':
        try:
            bgr = _enhance_mlkit_style(bgr)
        except Exception:
            pass
    if C.ENHANCE_MODE == 'bw':
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        bw = _binarize(gray)
        return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    return bgr


def _ocr_image(bgr: np.ndarray) -> Tuple[str, List[Dict[str, Any]]]:
    if not HAS_TESSERACT or not C.OCR_ENABLED:
        return "", []
    config = f"--oem {C.OCR_OEM} --psm {C.OCR_PSM}"
    try:
        text = pytesseract.image_to_string(bgr, lang=C.OCR_LANG, config=config)
    except Exception:
        text = ""
    try:
        data = pytesseract.image_to_data(bgr, lang=C.OCR_LANG, config=config, output_type=pytesseract.Output.DICT)
        n = len(data.get('text', []))
        words: List[Dict[str, Any]] = []
        for i in range(n):
            conf = int(data['conf'][i]) if str(data['conf'][i]).isdigit() else -1
            if conf < C.OCR_MIN_CONF:
                continue
            txt = data['text'][i].strip()
            if not txt:
                continue
            words.append({
                'text': txt,
                'conf': conf,
                'left': int(data['left'][i]),
                'top': int(data['top'][i]),
                'width': int(data['width'][i]),
                'height': int(data['height'][i]),
                'block_num': int(data.get('block_num', [0]*n)[i]),
                'par_num': int(data.get('par_num', [0]*n)[i]),
                'line_num': int(data.get('line_num', [0]*n)[i]),
                'word_num': int(data.get('word_num', [0]*n)[i]),
            })
    except Exception:
        words = []
    return text, words


def ocr_image_path(input_path: str, prefer_processed: bool = True) -> Dict[str, Any]:
    logger = logging.getLogger(__name__)
    if prefer_processed:
        cand = _prefer_processed_path(input_path)
        if cand:
            input_path = cand
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")
    bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Failed to read image")
    pre = _prepare_for_ocr(bgr)
    text, words = _ocr_image(pre)
    logger.info("ocr_done path=%s chars=%d words=%d", input_path, len(text), len(words))
    return {
        'text': text,
        'words': words,
        'language': C.OCR_LANG,
        'oem': C.OCR_OEM,
        'psm': C.OCR_PSM,
        'usedProcessed': bool(prefer_processed and _prefer_processed_path(input_path) is not None)
    }

