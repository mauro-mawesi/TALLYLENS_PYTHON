from typing import Tuple
import numpy as np
import cv2
from .geometry import corner_angles, parallelism_score, quad_dimensions


def calculate_score(corners: np.ndarray, img_shape: Tuple[int, int, int]) -> int:
    score = 0
    h, w = img_shape[:2]
    img_area = h * w

    # 1. Área
    quad_area = cv2.contourArea(corners.astype(np.float32))
    if img_area <= 0:
        return 0
    area_ratio = quad_area / img_area
    if 0.35 <= area_ratio <= 0.85:
        score += 40
    elif 0.20 <= area_ratio < 0.35:
        score += 30
    elif 0.85 < area_ratio <= 0.95:
        score += 25
    else:
        return 0

    # 2. Ángulos
    angles = corner_angles(corners)
    angle_score = 0
    for ang in angles:
        if 80 <= ang <= 100:
            angle_score += 5
        elif 70 <= ang <= 110:
            angle_score += 3
    score += min(angle_score, 20)

    # 3. Simetría + paralelismo
    sides = [
        np.linalg.norm(corners[1] - corners[0]),
        np.linalg.norm(corners[2] - corners[1]),
        np.linalg.norm(corners[3] - corners[2]),
        np.linalg.norm(corners[0] - corners[3]),
    ]
    top_bottom_ratio = min(sides[0], sides[2]) / max(sides[0], sides[2]) if max(sides[0], sides[2]) else 0
    left_right_ratio = min(sides[1], sides[3]) / max(sides[1], sides[3]) if max(sides[1], sides[3]) else 0
    parallel_bonus = parallelism_score(corners)  # 0..5
    if top_bottom_ratio > 0.90 and left_right_ratio > 0.90:
        score += 15 + parallel_bonus
    elif top_bottom_ratio > 0.80 and left_right_ratio > 0.80:
        score += 10 + parallel_bonus

    # 4. Convexidad
    if cv2.isContourConvex(corners.astype(np.float32)):
        score += 10

    # 5. Rectangularidad (área del quad vs rectángulo circunscrito)
    try:
        w, h = quad_dimensions(corners.astype(np.float32))
        rect_area = max(w * h, 1)
        quad_area = cv2.contourArea(corners.astype(np.float32))
        rect_ratio = quad_area / rect_area
        if rect_ratio > 0.92:
            score += 10
        elif rect_ratio > 0.85:
            score += 6
        elif rect_ratio > 0.75:
            score += 3
    except Exception:
        pass

    # 6. Distancia a bordes
    margin = 0.05
    corners_on_edge = 0
    for (x, y) in corners:
        if (x < w * margin or x > w * (1 - margin) or y < h * margin or y > h * (1 - margin)):
            corners_on_edge += 1
    if corners_on_edge == 0:
        score += 10
    elif corners_on_edge <= 2:
        score += 5

    return int(score)
