from typing import Tuple

import cv2

try:
    import pytesseract  # type: ignore
    HAS_TESSERACT = True
except Exception:
    HAS_TESSERACT = False

try:
    # Placeholder: orientation via doctr could be implemented here
    import doctr  # type: ignore
    HAS_DOCTR = True
except Exception:
    HAS_DOCTR = False


def detect_orientation(image_bgr) -> Tuple[int, str]:
    """
    Devuelve (rotation_angle, method)
    method in {"text_osd", "aspect_ratio", "none"}
    rotation_angle in {0, 90, 180, 270}
    """
    # Intentar con OSD de Tesseract si está disponible
    if HAS_TESSERACT:
        try:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            osd = pytesseract.image_to_osd(gray, output_type=pytesseract.Output.DICT)
            rot = int(osd.get("rotate", 0))
            # Normalizar a 0/90/180/270
            rot = ((rot % 360) + 360) % 360
            if rot in (0, 90, 180, 270):
                return rot, "text_osd"
        except Exception:
            pass

    # Fallback: aspect ratio
    h, w = image_bgr.shape[:2]
    ratio = w / max(h, 1)
    if ratio > 1.3:
        return 90, "aspect_ratio"
    return 0, "aspect_ratio"

