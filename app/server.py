from flask import Flask, request, jsonify
import os
import time
import logging
from werkzeug.utils import secure_filename

from .document_processor_hybrid import process_image_path
from .ocr_service import ocr_image_path

app = Flask(__name__)

# Logging básico configurable por env
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s %(levelname)s %(name)s %(message)s'
)

# Directorio para subidas (cuando se usa multipart)
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.route("/health", methods=["GET"])
def health():
    from .document_processor_hybrid import ML_DETECTOR
    from .ocr_service import HAS_TESSERACT as OCR_AVAILABLE
    ml_status = {
        "enabled": bool(os.getenv("USE_ML_DETECTOR", str(False)).lower() == "true" or False),
        "available": bool(ML_DETECTOR is not None and ML_DETECTOR.is_available()),
        "modelPath": os.getenv("ML_MODEL_PATH", "models/document_boundary.onnx"),
    }
    return jsonify({
        "status": "ok",
        "service": "document-processor",
        "version": "1.1.0",
        "ml": ml_status,
        "ocr": {"enabled": True, "available": bool(OCR_AVAILABLE)},
    }), 200


@app.route("/process-receipt", methods=["POST"])
def process_receipt():
    """Procesa una imagen vía:
    - multipart/form-data: campo 'image'
    - JSON: {'relativePath' | 'fileId' | 'fileName'}
    """
    saved_file_path = None

    # 1) Multipart file
    if request.files:
        image_file = request.files.get("image") or next(iter(request.files.values()), None)
        if image_file and image_file.filename:
            fname = secure_filename(image_file.filename)
            ts = int(time.time() * 1000)
            saved_file_path = os.path.join(UPLOAD_DIR, f"{ts}-{fname}")
            image_file.save(saved_file_path)
            logging.info(f"multipart_upload saved path={saved_file_path}")

    # 2) JSON con ruta
    path = saved_file_path
    if not path:
        data = request.get_json(force=True, silent=True) or {}
        file_id = data.get("fileId")
        relative_path = data.get("relativePath")
        file_name = data.get("fileName")  # opcional: nombre dentro de UPLOAD_DIR
        if not (relative_path or file_id or file_name):
            return jsonify({"error": "Missing 'relativePath' or 'fileId' or 'fileName' or 'image' file"}), 400

        raw_path = relative_path or file_id or file_name
        # Normalizar y evitar path traversal
        path = os.path.normpath(raw_path)
        if path.startswith(".."):
            return jsonify({"error": "Invalid path"}), 400

        # Resolver ruta contra UPLOAD_DIR si es relativa
        if not os.path.isabs(path):
            path = os.path.normpath(os.path.join(UPLOAD_DIR, path))
        else:
            # Remapeo opcional: si viene desde backend/uploads, mapear a UPLOAD_DIR
            backend_prefix = os.getenv("BACKEND_UPLOAD_PREFIX")
            if backend_prefix and path.startswith(backend_prefix):
                rel = os.path.relpath(path, backend_prefix)
                path = os.path.normpath(os.path.join(UPLOAD_DIR, rel))
        logging.info(f"resolved_input_path raw='{raw_path}' resolved='{path}' upload_dir='{UPLOAD_DIR}'")

    try:
        processed_path, result = process_image_path(path)
        resp = {
            "success": bool(result.get("processed", False)),
            "processed": bool(result.get("processed", False)),
            "processedPath": processed_path,
            "format": processed_path.split(".")[-1].lower(),
            "metadata": result.get("metadata", {}),
        }
        # Mantener 200 OK incluso si no se procesó, con success=false
        return jsonify(resp), 200
    except FileNotFoundError as e:
        logging.exception("file_not_found")
        return jsonify({"error": "Failed to process image", "details": str(e)}), 500
    except Exception as e:
        logging.exception("processing_error")
        return jsonify({"error": "Failed to process image", "details": str(e)}), 500


@app.route("/ocr-fallback", methods=["POST"])
def ocr_fallback():
    """OCR local sin costo (pytesseract). Acepta multipart 'image' o JSON con ruta.
    Parámetros opcionales (JSON):
      - preferProcessed: bool (true por defecto) → intenta usar *_processed primero
    """
    saved_file_path = None
    if request.files:
        image_file = request.files.get("image") or next(iter(request.files.values()), None)
        if image_file and image_file.filename:
            fname = secure_filename(image_file.filename)
            ts = int(time.time() * 1000)
            saved_file_path = os.path.join(UPLOAD_DIR, f"{ts}-{fname}")
            image_file.save(saved_file_path)
            logging.info(f"multipart_upload saved path={saved_file_path}")

    path = saved_file_path
    prefer_processed = True
    if not path:
        data = request.get_json(force=True, silent=True) or {}
        prefer_processed = bool(data.get("preferProcessed", True))
        file_id = data.get("fileId")
        relative_path = data.get("relativePath")
        file_name = data.get("fileName")
        if not (relative_path or file_id or file_name):
            return jsonify({"error": "Missing 'relativePath' or 'fileId' or 'fileName' or 'image' file"}), 400
        raw_path = relative_path or file_id or file_name
        path = os.path.normpath(raw_path)
        if path.startswith(".."):
            return jsonify({"error": "Invalid path"}), 400
        if not os.path.isabs(path):
            path = os.path.normpath(os.path.join(UPLOAD_DIR, path))
        else:
            backend_prefix = os.getenv("BACKEND_UPLOAD_PREFIX")
            if backend_prefix and path.startswith(backend_prefix):
                rel = os.path.relpath(path, backend_prefix)
                path = os.path.normpath(os.path.join(UPLOAD_DIR, rel))

    try:
        result = ocr_image_path(path, prefer_processed=prefer_processed)
        return jsonify({
            "success": True,
            "ocr": result
        }), 200
    except FileNotFoundError as e:
        logging.exception("ocr_file_not_found")
        return jsonify({"error": "Failed to OCR image", "details": str(e)}), 500
    except Exception as e:
        logging.exception("ocr_processing_error")
        return jsonify({"error": "Failed to OCR image", "details": str(e)}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
