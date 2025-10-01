from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import os
import time
import logging

import cv2
import numpy as np

from .config import ProcessingConfig as C

# Low-resource defaults: limitar hilos y desactivar OpenCL si disponible
try:
    cv2.setNumThreads(1)
except Exception:
    pass
try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass
from .utils.geometry import (
    order_corners,
    perspective_warp,
    line_intersection,
)
from .utils.scoring import calculate_score
from .utils.orientation import detect_orientation
from .ml.document_detector import DocumentBoundaryDetector


@dataclass
class DetectionResult:
    corners: Optional[np.ndarray]
    score: int
    method: str


def _resize_for_detection(img: np.ndarray) -> Tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    scale = 1.0
    max_dim = max(h, w)
    if max_dim > C.MAX_DIMENSION_DETECTION:
        scale = C.MAX_DIMENSION_DETECTION / max_dim
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img, scale


def _preprocess(gray: np.ndarray) -> np.ndarray:
    # Blur ligero y opcional CLAHE
    proc = cv2.GaussianBlur(gray, (5, 5), 0)
    if C.USE_CLAHE:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        proc = clahe.apply(proc)
    return proc


def _ensure_debug_dir() -> None:
    try:
        os.makedirs(C.DEBUG_DIR, exist_ok=True)
    except Exception:
        pass


def _debug_save(prefix: str, name: str, img: np.ndarray) -> None:
    if not C.SAVE_DEBUG_IMAGES:
        return
    try:
        _ensure_debug_dir()
        out = os.path.join(C.DEBUG_DIR, f"{prefix}_{name}.png")
        cv2.imwrite(out, img)
    except Exception:
        pass


def detect_adaptive(gray_img: np.ndarray) -> DetectionResult:
    thresh = cv2.adaptiveThreshold(
        gray_img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=C.ADAPTIVE_BLOCK_SIZE,
        C=C.ADAPTIVE_C,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours_out = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_out[-2]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    h, w = gray_img.shape[:2]
    img_area = h * w
    for cnt in contours[:5]:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > img_area * C.MIN_DOCUMENT_AREA_RATIO:
                corners = approx.reshape(4, 2)
                score = calculate_score(corners, gray_img.shape)
                return DetectionResult(corners=corners, score=score, method="adaptive_threshold")
    return DetectionResult(corners=None, score=0, method="adaptive_threshold")


def detect_hough(gray_img: np.ndarray) -> DetectionResult:
    edges = cv2.Canny(gray_img, C.CANNY_THRESHOLD1, C.CANNY_THRESHOLD2, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=C.HOUGH_THRESHOLD,
        minLineLength=C.HOUGH_MIN_LINE_LENGTH,
        maxLineGap=C.HOUGH_MAX_LINE_GAP,
    )
    if lines is None or len(lines) < 4:
        return DetectionResult(corners=None, score=0, method="canny_hough")

    # Clasificación básica por orientación (fallback simple)
    h_lines = []
    v_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angle = (angle + 180) % 180
        if angle < 20 or angle > 160:
            h_lines.append(line)
        elif 70 < angle < 110:
            v_lines.append(line)

    if len(h_lines) < 2 or len(v_lines) < 2:
        return DetectionResult(corners=None, score=0, method="canny_hough")

    top_line = min(h_lines, key=lambda l: min(l[0][1], l[0][3]))
    bottom_line = max(h_lines, key=lambda l: max(l[0][1], l[0][3]))
    left_line = min(v_lines, key=lambda l: min(l[0][0], l[0][2]))
    right_line = max(v_lines, key=lambda l: max(l[0][0], l[0][2]))

    corners = np.array([
        line_intersection(top_line, left_line),
        line_intersection(top_line, right_line),
        line_intersection(bottom_line, right_line),
        line_intersection(bottom_line, left_line),
    ], dtype=np.float32)

    score = calculate_score(corners, gray_img.shape)
    return DetectionResult(corners=corners, score=score, method="canny_hough")


def detect_doctr(text_img_bgr: np.ndarray) -> DetectionResult:
    """Fallback sin ML: aproximar hull de texto usando MSER + morfología.
    En producción, reemplazar por DocTR y aplicar hull+margen.
    """
    try:
        gray = cv2.cvtColor(text_img_bgr, cv2.COLOR_BGR2GRAY)
        mser = cv2.MSER_create(5, 100, 10000)
        regions, _ = mser.detectRegions(gray)
        if not regions:
            return DetectionResult(corners=None, score=0, method="doctr_fallback")
        # Combinar puntos de regiones
        pts = np.vstack([r for r in regions])
        hull = cv2.convexHull(pts)
        rect = cv2.minAreaRect(hull)
        box = cv2.boxPoints(rect)
        corners = np.array(box, dtype=np.float32)
        score = calculate_score(corners, gray.shape + (1,))
        return DetectionResult(corners=corners, score=score, method="doctr_fallback")
    except Exception:
        return DetectionResult(corners=None, score=0, method="doctr_fallback")


def detect_paper_color(bgr_img: np.ndarray) -> DetectionResult:
    """Detectar recibo por color/iluminancia (papel) en espacio Lab.
    Busca regiones altas en L y con a/b cercanos a neutro, luego aproxima a cuadrilátero.
    """
    try:
        lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        h, w = L.shape[:2]
        # Umbral dinámico por franja central (20% del ancho)
        band_w = max(int(w * 0.20), 10)
        x0 = (w - band_w) // 2
        central = L[:, x0:x0 + band_w]
        l_p50 = int(np.median(central))
        l_min = max(170, l_p50 - 5)
        a_tol = 22
        b_tol = 22
        mask_l = cv2.threshold(L, l_min, 255, cv2.THRESH_BINARY)[1]
        mask_a = cv2.inRange(A, 128 - a_tol, 128 + a_tol)
        mask_b = cv2.inRange(B, 128 - b_tol, 128 + b_tol)
        mask = cv2.bitwise_and(mask_l, cv2.bitwise_and(mask_a, mask_b))
        # Morfología con kernel elíptico: cerrar y luego abrir
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=1)

        contours_out = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_out[-2]
        if not contours:
            return DetectionResult(corners=None, score=0, method="paper_color")
        img_area = h * w
        # Tomar los contornos más grandes
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        best: Optional[DetectionResult] = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < img_area * C.MIN_DOCUMENT_AREA_RATIO:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) < 4:
                continue
            # Obtener caja mínima para estabilizar (preferir 4 puntos)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect).astype(np.float32)
            base_score = calculate_score(box, bgr_img.shape)
            # Bonus por blancura y baja cromaticidad dentro del quad
            try:
                poly = box.astype(np.int32)
                mask_poly = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask_poly, [poly], 255)
                l_mean = float(cv2.mean(L, mask=mask_poly)[0])
                a_mean = float(cv2.mean(A, mask=mask_poly)[0])
                b_mean = float(cv2.mean(B, mask=mask_poly)[0])
                chroma_dev = abs(a_mean - 128.0) + abs(b_mean - 128.0)
                white_bonus = 0
                if l_mean > 200 and chroma_dev < 18:
                    white_bonus = 20
                elif l_mean > 190 and chroma_dev < 24:
                    white_bonus = 12
                elif l_mean > 180 and chroma_dev < 30:
                    white_bonus = 6
                sc = int(base_score + white_bonus)
            except Exception:
                sc = int(base_score)
            cand = DetectionResult(corners=box, score=sc, method="paper_color")
            if (best is None) or (cand.score > best.score):
                best = cand
        if best is None:
            return DetectionResult(corners=None, score=0, method="paper_color")
        return best
    except Exception:
        return DetectionResult(corners=None, score=0, method="paper_color")


def detect_lsd_lines(gray_img: np.ndarray) -> DetectionResult:
    """Líneas por LSD (ximgproc) y cuadrilátero por intersecciones extremas.
    Más robusto que Hough en fondos con textura.
    """
    try:
        if not hasattr(cv2, 'ximgproc'):
            return DetectionResult(corners=None, score=0, method="lsd_lines")
        fld = cv2.ximgproc.createFastLineDetector()
        lines = fld.detect(gray_img)
        if lines is None or len(lines) < 4:
            return DetectionResult(corners=None, score=0, method="lsd_lines")
        # lines: Nx1x4 -> convertir a formato similar a Hough
        arr = []
        for l in lines:
            x1, y1, x2, y2 = l[0]
            arr.append([[int(x1), int(y1), int(x2), int(y2)]])
        lines_h = np.array(arr)

        # Clasificación por orientación (rangos más estrictos)
        h_lines = []
        v_lines = []
        for line in lines_h:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angle = (angle + 180) % 180
            if angle < 15 or angle > 165:
                h_lines.append(line)
            elif 75 < angle < 105:
                v_lines.append(line)
        if len(h_lines) < 2 or len(v_lines) < 2:
            return DetectionResult(corners=None, score=0, method="lsd_lines")

        # Elegir extremos por proyección: mín/max Y para horizontales, mín/max X para verticales
        def min_y(l):
            x1, y1, x2, y2 = l[0]; return min(y1, y2)
        def max_y(l):
            x1, y1, x2, y2 = l[0]; return max(y1, y2)
        def min_x(l):
            x1, y1, x2, y2 = l[0]; return min(x1, x2)
        def max_x(l):
            x1, y1, x2, y2 = l[0]; return max(x1, x2)
        top_line = min(h_lines, key=min_y)
        bottom_line = max(h_lines, key=max_y)
        left_line = min(v_lines, key=min_x)
        right_line = max(v_lines, key=max_x)

        corners = np.array([
            line_intersection(top_line, left_line),
            line_intersection(top_line, right_line),
            line_intersection(bottom_line, right_line),
            line_intersection(bottom_line, left_line),
        ], dtype=np.float32)
        score = calculate_score(corners, gray_img.shape)
        return DetectionResult(corners=corners, score=score, method="lsd_lines")
    except Exception:
        return DetectionResult(corners=None, score=0, method="lsd_lines")


def detect_grabcut_center(bgr_img: np.ndarray) -> DetectionResult:
    """Fallback robusto: semilla foreground en banda central y background en bordes.
    Útil para recibos centrados sobre fondo con textura.
    """
    try:
        h, w = bgr_img.shape[:2]
        if h < 50 or w < 50:
            return DetectionResult(corners=None, score=0, method="grabcut_center")
        mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
        # Semillas FG: franja central del 20% del ancho
        band_w = max(int(w * 0.20), 10)
        x0 = (w - band_w) // 2
        mask[:, x0:x0 + band_w] = cv2.GC_PR_FGD
        # Semillas BG: bordes 6% por lado
        m = max(int(w * 0.06), 4)
        mask[:, :m] = cv2.GC_BGD
        mask[:, w - m:] = cv2.GC_BGD
        n = max(int(h * 0.06), 4)
        mask[:n, :] = cv2.GC_BGD
        mask[h - n:, :] = cv2.GC_BGD
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        cv2.grabCut(bgr_img, mask, None, bgdModel, fgdModel, 2, cv2.GC_INIT_WITH_MASK)
        mask2 = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')
        contours_out = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = contours_out[-2]
        if not cnts:
            return DetectionResult(corners=None, score=0, method="grabcut_center")
        cnt = max(cnts, key=cv2.contourArea)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect).astype(np.float32)
        score = calculate_score(box, bgr_img.shape)
        return DetectionResult(corners=box, score=score, method="grabcut_center")
    except Exception:
        return DetectionResult(corners=None, score=0, method="grabcut_center")


def postprocess_output(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    max_dim = max(h, w)
    if max_dim > C.MAX_DIMENSION_OUTPUT:
        scale = C.MAX_DIMENSION_OUTPUT / max_dim
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if C.OUTPUT_COLORSPACE.upper() == 'GRAY' and len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def process_image_path(input_path: str) -> Tuple[str, Dict[str, Any]]:
    logger = logging.getLogger(__name__)
    t0 = time.time()
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Failed to read image")

    orig_h, orig_w = bgr.shape[:2]
    logger.info("processing_start path=%s width=%d height=%d", input_path, orig_w, orig_h)
    # Prefijo para debug
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    debug_prefix = f"{base_name}_{int(t0*1000)}"
    resized, scale = _resize_for_detection(bgr)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = _preprocess(gray)
    _debug_save(debug_prefix, "resized", resized)
    _debug_save(debug_prefix, "gray", gray)

    # Ejecutar métodos según orden
    results: list[DetectionResult] = []
    for method in C.METHOD_ORDER:
        try:
            if method == "ml_detector":
                res = detect_ml(resized)
            elif method == "adaptive_threshold":
                res = detect_adaptive(gray)
            elif method == "paper_color":
                res = detect_paper_color(resized)
            elif method == "grabcut_center":
                res = detect_grabcut_center(resized)
            elif method == "lsd_lines":
                res = detect_lsd_lines(gray)
            elif method == "canny_hough":
                res = detect_hough(gray)
            elif method == "doctr_fallback":
                res = detect_doctr(resized)
            else:
                continue
            results.append(res)
        except Exception:
            # Ignorar fallo de método pero continuar
            continue

    for r in results:
        logger.info("detection_result method=%s score=%d hasCorners=%s", r.method, int(r.score), bool(r.corners is not None))

    # Seleccionar mejor
    best = max(results, key=lambda r: r.score if r.corners is not None else -1, default=DetectionResult(None, 0, "none"))

    applied_warp = False
    rotation_applied = False
    rotation_angle = 0
    output = bgr.copy()
    kept_area_percent = 100.0
    action = "none"

    if best.corners is not None:
        # Reescalar esquinas a tamaño original
        corners_orig = (best.corners / max(scale, 1e-8)).astype(np.float32)
        # Validar aspecto plausible del recibo
        try:
            rect_aspect = order_corners(corners_orig)
            wq = float(np.linalg.norm(rect_aspect[1] - rect_aspect[0]))
            hq = float(np.linalg.norm(rect_aspect[3] - rect_aspect[0]))
            aspect = max(wq, hq) / max(min(wq, hq), 1e-6)
            if aspect < C.DOC_ASPECT_MIN or aspect > C.DOC_ASPECT_MAX:
                corners_orig = None
        except Exception:
            pass

        # Refinar con GrabCut si está habilitado
        if corners_orig is not None and C.ENABLE_GRABCUT_REFINE:
            refined = _refine_corners_grabcut(bgr, corners_orig)
            if refined is not None:
                corners_orig = refined
                best.score = max(best.score, calculate_score(corners_orig, bgr.shape))

        if corners_orig is not None and best.score >= C.MIN_SCORE_FOR_WARP:
            # Chequeo de rectangularidad para evitar warps malos
            try:
                rect = order_corners(corners_orig)
                wq, hq = int(np.linalg.norm(rect[1]-rect[0])), int(np.linalg.norm(rect[3]-rect[0]))
                rect_area = max(wq*hq, 1)
                quad_area = cv2.contourArea(rect.astype(np.float32))
                rect_ratio = quad_area / rect_area
            except Exception:
                rect_ratio = 1.0
            # Área del quad vs imagen para evitar warps sin valor
            img_area = orig_h * orig_w
            quad_img_ratio = float(cv2.contourArea(corners_orig.astype(np.float32)) / max(img_area, 1))
            if (rect_ratio >= C.RECTANGULARITY_MIN_FOR_WARP and
                C.QUAD_AREA_MIN_RATIO <= quad_img_ratio <= C.QUAD_AREA_MAX_RATIO):
                output = perspective_warp(bgr, corners_orig)
                applied_warp = True
                action = "warp"
                kept_area_percent = (output.shape[0] * output.shape[1]) / (orig_h * orig_w) * 100.0
            else:
                # Si la rectangularidad es baja, degradar a crop seguro
                rect = order_corners(corners_orig)
                x_min = max(int(min(rect[:, 0])), 0)
                y_min = max(int(min(rect[:, 1])), 0)
                x_max = min(int(max(rect[:, 0])), orig_w)
                y_max = min(int(max(rect[:, 1])), orig_h)
                # Expandir con margen seguro
                mw = int((x_max - x_min) * C.SAFE_MARGIN_PERCENT)
                mh = int((y_max - y_min) * C.SAFE_MARGIN_PERCENT * 1.5)
                x_min = max(x_min - mw, 0)
                x_max = min(x_max + mw, orig_w)
                y_min = max(y_min - mh, 0)
                y_max = min(y_max + mh, orig_h)
                output = bgr[y_min:y_max, x_min:x_max]
                action = "crop"
                kept_area_percent = (output.shape[0] * output.shape[1]) / (orig_h * orig_w) * 100.0
        elif corners_orig is not None and best.score >= C.MIN_SCORE_FOR_CROP:
            rect = order_corners(corners_orig)
            x_min = max(int(min(rect[:, 0])), 0)
            y_min = max(int(min(rect[:, 1])), 0)
            x_max = min(int(max(rect[:, 0])), orig_w)
            y_max = min(int(max(rect[:, 1])), orig_h)
            # Expandir con margen seguro
            mw = int((x_max - x_min) * C.SAFE_MARGIN_PERCENT)
            mh = int((y_max - y_min) * C.SAFE_MARGIN_PERCENT * 1.5)
            x_min = max(x_min - mw, 0)
            x_max = min(x_max + mw, orig_w)
            y_min = max(y_min - mh, 0)
            y_max = min(y_max + mh, orig_h)
            output = bgr[y_min:y_max, x_min:x_max]
            action = "crop"
            kept_area_percent = (output.shape[0] * output.shape[1]) / (orig_h * orig_w) * 100.0
        # else: dejar imagen original

        # Visualización de detección en el espacio redimensionado
        try:
            vis = resized.copy()
            cv2.polylines(vis, [best.corners.astype(np.int32)], True, (0, 255, 0), 2)
            _debug_save(debug_prefix, f"detection_{best.method}", vis)
        except Exception:
            pass

    # Orientación
    rot, orient_method = detect_orientation(output)
    if rot in (90, 180, 270):
        rotation_applied = True
        rotation_angle = rot
        if action == "none":
            action = "rotate"
        # cv2.rotate usa códigos para 90/180/270
        if rot == 90:
            output = cv2.rotate(output, cv2.ROTATE_90_CLOCKWISE)
        elif rot == 180:
            output = cv2.rotate(output, cv2.ROTATE_180)
        elif rot == 270:
            output = cv2.rotate(output, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        orient_method = orient_method or "none"

    # Mejora tipo MLKit
    if C.ENHANCE_PROFILE == 'mlkit' and C.ENHANCE_MODE != 'none':
        try:
            output = _enhance_mlkit_style(output)
        except Exception:
            pass

    # Margen blanco alrededor
    try:
        output = _pad_border(output, C.PAD_BORDER_PERCENT)
    except Exception:
        pass

    output = postprocess_output(output)
    # Trim adicional sensible a contenido (muy conservador)
    try:
        output = _content_safe_trim(output, max_trim_y_frac=C.TRIM_MAX_FRAC_Y, max_trim_x_frac=C.TRIM_MAX_FRAC_X)
    except Exception:
        pass
    # Recorte de bordes uniformes remanentes tras warp/crop (muy conservador en vertical)
    if C.ENABLE_TRIM:
        try:
            output = _trim_uniform_borders(output, C.TRIM_TOL, C.TRIM_MAX_FRAC_X, C.TRIM_MAX_FRAC_Y)
        except Exception:
            pass

    # Guardado: evitar bloat si no hay cambios significativos
    base, ext = os.path.splitext(input_path)
    selected_format = C.OUTPUT_FORMAT.lower()
    # Si no hubo warp/crop y no hubo rotación, no re-escribir: devolver original
    if (not applied_warp) and (best.corners is None or best.score < C.MIN_SCORE_FOR_CROP) and (not rotation_applied):
        dt = int((time.time() - t0) * 1000)
        processed = False
        # Si se solicita, escribir copia _processed para inspección
        if C.ALWAYS_WRITE_PROCESSED:
            base, _ = os.path.splitext(input_path)
            selected_format = C.OUTPUT_FORMAT.lower()
            ext_out = 'png' if selected_format == 'png' else 'webp'
            out_path = f"{base}_processed.{ext_out}"
            encode_params = []
            if selected_format == 'png':
                encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 0]
            else:
                # WebP
                quality = 90 if selected_format == 'webp' else C.OUTPUT_QUALITY
                encode_params = [cv2.IMWRITE_WEBP_QUALITY, quality]
            cv2.imwrite(out_path, output, encode_params)
            _debug_save(debug_prefix, "output", output)
            out_name = os.path.basename(out_path)
        else:
            out_name = os.path.basename(input_path)

        metadata: Dict[str, Any] = {
            "originalDimensions": {"width": orig_w, "height": orig_h},
            "processedDimensions": {"width": int(output.shape[1]), "height": int(output.shape[0])},
            "detectionMethod": best.method if best.corners is not None else "none",
            "detectionScore": int(best.score if best.corners is not None else 0),
            "perspectiveCorrectionApplied": bool(applied_warp),
            "rotationApplied": bool(rotation_applied),
            "rotationAngle": int(rotation_angle),
            "keptAreaPercent": float(round(kept_area_percent, 2)),
            "orientationMethod": orient_method,
            "colorSpace": "GRAY" if (len(output.shape) == 2) else "RGB",
            "processingTimeMs": dt,
            "savedCopy": bool(C.ALWAYS_WRITE_PROCESSED),
            "processingAction": action,
            "enhanceProfile": C.ENHANCE_PROFILE,
            "enhanceMode": C.ENHANCE_MODE,
        }
        logger.info("processing_done path=%s method=%s score=%d processed=%s keptArea=%.2f%% timeMs=%d out=%s", input_path, metadata["detectionMethod"], metadata["detectionScore"], bool(processed), metadata["keptAreaPercent"], dt, out_name)
        return out_name, {"processed": processed, "metadata": metadata}

    # Si solo hubo rotación, usar WebP con calidad 90 para tamaño razonable salvo que se pida explícito
    if (not applied_warp) and (best.corners is None or best.score < C.MIN_SCORE_FOR_CROP) and rotation_applied:
        selected_format = 'webp' if selected_format == 'png' else selected_format

    ext_out = 'png' if selected_format == 'png' else 'webp'
    out_path = f"{base}_processed.{ext_out}"
    # Parámetros
    encode_params = []
    if selected_format == 'png':
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 0]
    elif selected_format == 'webp' or selected_format.startswith('webp'):
        quality = 90 if selected_format == 'webp' else C.OUTPUT_QUALITY
        encode_params = [cv2.IMWRITE_WEBP_QUALITY, quality]
    cv2.imwrite(out_path, output, encode_params)
    _debug_save(debug_prefix, "output", output)

    dt = int((time.time() - t0) * 1000)

    metadata: Dict[str, Any] = {
        "originalDimensions": {"width": orig_w, "height": orig_h},
        "processedDimensions": {"width": int(output.shape[1]), "height": int(output.shape[0])},
        "detectionMethod": best.method if best.corners is not None else "none",
        "detectionScore": int(best.score if best.corners is not None else 0),
        "perspectiveCorrectionApplied": bool(applied_warp),
        "rotationApplied": bool(rotation_applied),
        "rotationAngle": int(rotation_angle),
        "keptAreaPercent": float(round(kept_area_percent, 2)),
        "orientationMethod": orient_method,
        "colorSpace": "GRAY" if (len(output.shape) == 2) else "RGB",
        "processingTimeMs": dt,
        "processingAction": action,
        "enhanceProfile": C.ENHANCE_PROFILE,
        "enhanceMode": C.ENHANCE_MODE,
    }

    processed = applied_warp or (best.corners is not None and best.score >= C.MIN_SCORE_FOR_CROP) or rotation_applied
    logger.info(
        "processing_done path=%s method=%s score=%d processed=%s keptArea=%.2f%% timeMs=%d out=%s",
        input_path,
        metadata["detectionMethod"],
        metadata["detectionScore"],
        bool(processed),
        metadata["keptAreaPercent"],
        dt,
        out_path,
    )

    # Devolver solo el nombre del archivo procesado (no la ruta completa)
    out_name = os.path.basename(out_path)
    return out_name, {"processed": processed, "metadata": metadata}
def _trim_uniform_borders(img: np.ndarray, tol: int, max_frac_x: float, max_frac_y: float) -> np.ndarray:
    """Recorta bordes uniformes casi blancos/constantes post-warp.
    tol: tolerancia por canal (0..255)
    max_frac_x/max_frac_y: límites por eje para evitar cortar contenido vertical.
    """
    h, w = img.shape[:2]
    max_cut_x = int(w * max_frac_x)
    max_cut_y = int(h * max_frac_y)
    def is_uniform_line(arr):
        mn = arr.min(axis=0).astype(int)
        mx = arr.max(axis=0).astype(int)
        return bool(np.all((mx - mn) <= tol))
    top = 0
    while top < max_cut_y and is_uniform_line(img[top:top+1, :]):
        top += 1
    bottom = h
    while bottom - 1 > h - max_cut_y and is_uniform_line(img[bottom-1:bottom, :]):
        bottom -= 1
    left = 0
    while left < max_cut_x and is_uniform_line(img[:, left:left+1]):
        left += 1
    right = w
    while right - 1 > w - max_cut_x and is_uniform_line(img[:, right-1:right]):
        right -= 1
    if top < bottom and left < right:
        return img[top:bottom, left:right]
    return img
def _refine_corners_grabcut(bgr: np.ndarray, corners: np.ndarray, iters: int = 2) -> Optional[np.ndarray]:
    try:
        h, w = bgr.shape[:2]
        rect = order_corners(corners.astype(np.float32))
        x_min = max(int(rect[:, 0].min()), 0)
        y_min = max(int(rect[:, 1].min()), 0)
        x_max = min(int(rect[:, 0].max()), w - 1)
        y_max = min(int(rect[:, 1].max()), h - 1)
        bw = x_max - x_min
        bh = y_max - y_min
        if bw < 10 or bh < 10:
            return None
        pad = 10
        x0 = max(x_min - pad, 0)
        y0 = max(y_min - pad, 0)
        x1 = min(x_max + pad, w - 1)
        y1 = min(y_max + pad, h - 1)
        roi = bgr[y0:y1, x0:x1].copy()
        mask = np.zeros(roi.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        cv2.grabCut(roi, mask, (pad, pad, bw, bh), bgdModel, fgdModel, iters, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
        contours_out = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = contours_out[-2]
        if not cnts:
            return None
        cnt = max(cnts, key=cv2.contourArea)
        rect2 = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect2)
        box[:, 0] += x0
        box[:, 1] += y0
        return box.astype(np.float32)
    except Exception:
        return None


def _white_balance_grayworld(bgr: np.ndarray) -> np.ndarray:
    try:
        result = bgr.astype(np.float32)
        avg_b = np.mean(result[:, :, 0]) + 1e-6
        avg_g = np.mean(result[:, :, 1]) + 1e-6
        avg_r = np.mean(result[:, :, 2]) + 1e-6
        gray = (avg_b + avg_g + avg_r) / 3.0
        result[:, :, 0] *= (gray / avg_b)
        result[:, :, 1] *= (gray / avg_g)
        result[:, :, 2] *= (gray / avg_r)
        return np.clip(result, 0, 255).astype(np.uint8)
    except Exception:
        return bgr


def _enhance_mlkit_style(bgr: np.ndarray) -> np.ndarray:
    """Intento de aproximación al estilo MLKit: balance de blancos, normalización de iluminación,
    aumento de contraste (CLAHE) y enfoque suave; opcional B/N con umbral adaptativo.
    """
    try:
        img = _white_balance_grayworld(bgr)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        # Normalización de iluminación (estimación suave del campo de iluminación)
        ksize = max(int(min(img.shape[:2]) * 0.02) | 1, 3)
        illum = cv2.GaussianBlur(L, (0, 0), ksize)
        L_norm = cv2.normalize(cv2.divide(L, illum, scale=128), None, 0, 255, cv2.NORM_MINMAX)
        # Contraste (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=C.CLAHE_CLIP, tileGridSize=(C.CLAHE_TILE, C.CLAHE_TILE))
        L_eq = clahe.apply(L_norm.astype(np.uint8))

        if C.ENHANCE_MODE == 'bw':
            # Salida B/N nítida
            thr = cv2.adaptiveThreshold(L_eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10)
            return cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR)

        # Reconstruir color con L mejorada
        lab2 = cv2.merge((L_eq, A, B))
        out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
        # Suavizado ligero + unsharp mask
        out_sm = cv2.bilateralFilter(out, d=5, sigmaColor=50, sigmaSpace=50)
        blur = cv2.GaussianBlur(out_sm, (0, 0), 1.0)
        out_sharp = cv2.addWeighted(out_sm, 1.2, blur, -0.2, 0)
        return out_sharp
    except Exception:
        return bgr


def _pad_border(img: np.ndarray, percent: float) -> np.ndarray:
    try:
        if percent <= 0:
            return img
        h, w = img.shape[:2]
        pad_y = int(h * percent)
        pad_x = int(w * percent)
        color = (255, 255, 255)
        return cv2.copyMakeBorder(img, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=color)
    except Exception:
        return img


def _content_safe_trim(img: np.ndarray, max_trim_y_frac: float = 0.02, max_trim_x_frac: float = 0.02) -> np.ndarray:
    """Recorta bandas vacías usando energía de bordes para no cortar texto.
    Solo recorta si la energía (Sobel) en el borde es muy baja y sube claramente tras unas filas/columnas.
    """
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        h, w = gray.shape[:2]
        max_trim_y = max(1, int(h * max_trim_y_frac))
        max_trim_x = max(1, int(w * max_trim_x_frac))
        # Energía por fila/columna
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        row_energy = np.mean(mag, axis=1)
        col_energy = np.mean(mag, axis=0)
        # Umbral dinámico
        thr_r = np.percentile(row_energy, 35)
        thr_c = np.percentile(col_energy, 35)
        top = 0
        while top < max_trim_y and row_energy[top] < thr_r * 0.5:
            # detener si siguiente ventana no sube energía
            if top + 3 < h and np.mean(row_energy[top:top+3]) >= thr_r * 0.6:
                break
            top += 1
        bottom = h
        while (h - bottom) < max_trim_y and row_energy[bottom-1] < thr_r * 0.5:
            if bottom - 3 >= 0 and np.mean(row_energy[bottom-3:bottom]) >= thr_r * 0.6:
                break
            bottom -= 1
        left = 0
        while left < max_trim_x and col_energy[left] < thr_c * 0.5:
            if left + 3 < w and np.mean(col_energy[left:left+3]) >= thr_c * 0.6:
                break
            left += 1
        right = w
        while (w - right) < max_trim_x and col_energy[right-1] < thr_c * 0.5:
            if right - 3 >= 0 and np.mean(col_energy[right-3:right]) >= thr_c * 0.6:
                break
            right -= 1
        if 0 <= top < bottom <= h and 0 <= left < right <= w:
            return img[top:bottom, left:right]
        return img
    except Exception:
        return img
ML_DETECTOR = DocumentBoundaryDetector(C.ML_MODEL_PATH) if C.USE_ML_DETECTOR else None


def detect_ml(bgr_img: np.ndarray) -> DetectionResult:
    try:
        if ML_DETECTOR is None or not ML_DETECTOR.is_available():
            return DetectionResult(corners=None, score=0, method="ml_detector")
        poly = ML_DETECTOR.predict(bgr_img)
        if poly is None:
            return DetectionResult(corners=None, score=0, method="ml_detector")
        score = calculate_score(poly.astype(np.float32), bgr_img.shape)
        return DetectionResult(corners=poly.astype(np.float32), score=score, method="ml_detector")
    except Exception:
        return DetectionResult(corners=None, score=0, method="ml_detector")
