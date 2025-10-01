import os
from typing import Optional, Tuple

import numpy as np

try:
    import onnxruntime as ort  # type: ignore
    HAS_ORT = True
except Exception:
    HAS_ORT = False


class DocumentBoundaryDetector:
    """
    Wrapper para un modelo ONNX de detección/segmentación de documentos.
    Espera un modelo que, dado una imagen, devuelva un polígono de 4 puntos (x, y) en coords de entrada.

    Nota: Esta clase es un stub profesional listo para conectar tu modelo.
    Implementa pre/post-proceso genéricos y devuelve None si el modelo no está disponible.
    """

    def __init__(self, model_path: str, providers: Optional[list[str]] = None) -> None:
        self.available = False
        self.session = None
        self.input_name = None
        self.input_shape = None  # (N, C, H, W)
        self.model_path = model_path
        if HAS_ORT and os.path.isfile(model_path):
            prov = providers or ["CPUExecutionProvider"]
            try:
                self.session = ort.InferenceSession(model_path, providers=prov)
                self.input_name = self.session.get_inputs()[0].name
                ishape = self.session.get_inputs()[0].shape
                # Permitir dynamic axes; guardar H, W si están definidos
                self.input_shape = ishape
                self.available = True
            except Exception:
                self.available = False

    def is_available(self) -> bool:
        return bool(self.available)

    def _preprocess(self, bgr: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        # Normalizar a tamaño de entrada si está definido, si no usar 640
        h, w = bgr.shape[:2]
        target = 640
        if self.input_shape and len(self.input_shape) == 4 and isinstance(self.input_shape[2], int) and isinstance(self.input_shape[3], int):
            target = int(max(self.input_shape[2], self.input_shape[3]))
        scale = target / max(h, w)
        if scale != 1.0:
            nh, nw = int(h * scale), int(w * scale)
            import cv2
            bgr_resized = cv2.resize(bgr, (nw, nh))
        else:
            bgr_resized = bgr
        # Letterbox a cuadrado target x target
        pad_h = target - bgr_resized.shape[0]
        pad_w = target - bgr_resized.shape[1]
        top, left = pad_h // 2, pad_w // 2
        import cv2
        padded = cv2.copyMakeBorder(bgr_resized, top, pad_h - top, left, pad_w - left, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        # BGR -> RGB, HWC->CHW, float32 [0,1]
        rgb = padded[:, :, ::-1].astype(np.float32) / 255.0
        chw = np.transpose(rgb, (0, 1, 2))
        chw = np.transpose(chw, (2, 0, 1))[None, ...]
        return chw, scale, (top, left)

    def _postprocess_polygon(self, poly_norm: np.ndarray, scale: float, pad: Tuple[int, int], orig_hw: Tuple[int, int]) -> np.ndarray:
        # poly_norm: (4, 2) en coords de la imagen padded (target x target)
        top, left = pad
        h, w = orig_hw
        # Revertir letterbox y escala
        pts = poly_norm.copy().astype(np.float32)
        pts[:, 0] -= left
        pts[:, 1] -= top
        pts /= max(scale, 1e-8)
        # Recortar a imagen
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        return pts

    def predict(self, bgr: np.ndarray) -> Optional[np.ndarray]:
        if not self.available:
            return None
        try:
            inp, scale, pad = self._preprocess(bgr)
            outputs = self.session.run(None, {self.input_name: inp})
            # Esperamos que el modelo devuelva un polígono (4,2) en coords del input padded
            poly = None
            for out in outputs:
                if isinstance(out, np.ndarray) and out.shape[-2:] == (4, 2):
                    poly = out.reshape(4, 2)
                    break
            if poly is None:
                return None
            return self._postprocess_polygon(poly, scale, pad, bgr.shape[:2])
        except Exception:
            return None

