from typing import List, Tuple
import numpy as np
import cv2


def order_corners(pts: np.ndarray) -> np.ndarray:
    # pts: (4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]  # TL
    rect[2] = pts[np.argmax(s)]  # BR
    rect[1] = pts[np.argmin(diff)]  # TR
    rect[3] = pts[np.argmax(diff)]  # BL
    return rect


def quad_dimensions(rect: np.ndarray) -> Tuple[int, int]:
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxWidth = int(max(widthA, widthB))
    maxHeight = int(max(heightA, heightB))
    return maxWidth, maxHeight


def perspective_warp(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    rect = order_corners(corners.astype(np.float32))
    maxWidth, maxHeight = quad_dimensions(rect)
    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    # Heurística de interpolación: CUBIC si amplía, AREA si reduce
    h, w = image.shape[:2]
    scale_w = maxWidth / max(w, 1)
    scale_h = maxHeight / max(h, 1)
    enlarging = scale_w > 1.0 or scale_h > 1.0
    flags = cv2.INTER_CUBIC if enlarging else cv2.INTER_AREA
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=flags)
    return warped


def line_intersection(l1: np.ndarray, l2: np.ndarray) -> Tuple[float, float]:
    x1, y1, x2, y2 = l1[0]
    x3, y3, x4, y4 = l2[0]
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return (0.0, 0.0)
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return (px, py)


def corner_angles(corners: np.ndarray) -> List[float]:
    # corners en orden cualquiera
    rect = order_corners(corners)
    angles = []
    for i in range(4):
        p0 = rect[i]
        p1 = rect[(i - 1) % 4]
        p2 = rect[(i + 1) % 4]
        v1 = p1 - p0
        v2 = p2 - p0
        a1 = np.arctan2(v1[1], v1[0])
        a2 = np.arctan2(v2[1], v2[0])
        ang = np.abs((a2 - a1) * 180 / np.pi)
        if ang > 180:
            ang = 360 - ang
        angles.append(ang)
    return angles


def parallelism_score(corners: np.ndarray) -> int:
    rect = order_corners(corners)
    # vectores de lados opuestos
    top = rect[1] - rect[0]
    bottom = rect[2] - rect[3]
    left = rect[3] - rect[0]
    right = rect[2] - rect[1]
    def cos_sim(a, b):
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))
    # cercanía a paralelos: |cos| cerca de 1
    tb = abs(cos_sim(top, bottom))
    lr = abs(cos_sim(left, right))
    avg = (tb + lr) / 2.0
    if avg > 0.98:
        return 5
    if avg > 0.95:
        return 4
    if avg > 0.90:
        return 3
    if avg > 0.85:
        return 2
    if avg > 0.80:
        return 1
    return 0

