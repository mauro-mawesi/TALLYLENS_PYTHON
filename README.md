# TallyLens Document Processor

Python-based document processing microservice for receipt image enhancement, text detection, and OCR optimization using OpenCV and computer vision techniques.

## Features

- **Document Detection**: Multi-method document boundary detection (color-based, GrabCut, LSD lines, adaptive thresholding)
- **Perspective Correction**: Automatic perspective warp for skewed documents
- **Image Enhancement**: MLKit-style image enhancement with contrast adjustment, denoising, and sharpening
- **Orientation Detection**: Automatic text orientation detection and correction
- **Border Trimming**: Smart border removal with content preservation
- **OCR Integration**: Local Tesseract OCR support with configurable parameters
- **ML Support**: Optional ONNX model integration for neural network-based document detection
- **RESTful API**: Flask-based API for easy integration

## Tech Stack

- **Language**: Python 3.9+
- **Framework**: Flask
- **Computer Vision**: OpenCV (cv2)
- **OCR**: Tesseract OCR (pytesseract)
- **Image Processing**: NumPy, Pillow
- **Document Detection**: DocTR (optional)
- **ML Inference**: ONNX Runtime (optional)

## Prerequisites

- Python 3.9+
- Tesseract OCR 4.0+
- OpenCV dependencies
- (Optional) ONNX Runtime for ML-based detection

## Quick Start

### 1. Clone and Install

```bash
git clone git@github.com:mauro-mawesi/TALLYLENS_PYTHON.git
cd TALLYLENS_PYTHON
pip install -r requirements.txt
```

### 2. Install System Dependencies

#### Ubuntu/Debian

```bash
sudo apt-get update
sudo apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-spa \
    tesseract-ocr-nld \
    libgl1-mesa-glx \
    libglib2.0-0
```

#### macOS

```bash
brew install tesseract
```

#### Windows

Download and install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki

### 3. Configure Environment

Copy the example environment file and configure it:

```bash
cp .env.example .env
```

Edit `.env` with your configuration:

```env
# Server
HOST=0.0.0.0
PORT=5000
LOG_LEVEL=INFO

# Upload directory
UPLOAD_DIR=uploads

# OCR Configuration
OCR_LANG=eng+spa+nld
OCR_OEM=3
OCR_PSM=6
OCR_MIN_CONF=40

# ML Detector (optional)
USE_ML_DETECTOR=false
ML_MODEL_PATH=models/document_boundary.onnx

# Backend integration (optional)
BACKEND_UPLOAD_PREFIX=/path/to/backend/uploads
```

### 4. Run the Service

```bash
# Development
python -m app.server

# With Flask
export FLASK_APP=app.server
flask run --host=0.0.0.0 --port=5000

# Production (with gunicorn)
gunicorn -w 4 -b 0.0.0.0:5000 app.server:app
```

The service will be available at `http://localhost:5000`

## API Endpoints

### Health Check

```bash
GET /health

Response:
{
  "status": "ok",
  "service": "document-processor",
  "version": "1.1.0",
  "ml": {
    "enabled": false,
    "available": false,
    "modelPath": "models/document_boundary.onnx"
  },
  "ocr": {
    "enabled": true,
    "available": true
  }
}
```

### Process Receipt

```bash
POST /process-receipt

# Option 1: Multipart file upload
curl -X POST http://localhost:5000/process-receipt \
  -F "image=@receipt.jpg"

# Option 2: JSON with file path
curl -X POST http://localhost:5000/process-receipt \
  -H "Content-Type: application/json" \
  -d '{"relativePath": "receipt.jpg"}'

Response:
{
  "success": true,
  "processed": true,
  "processedPath": "uploads/receipt_processed.webp",
  "format": "webp",
  "metadata": {
    "method": "paper_color",
    "score": 95,
    "orientation": "portrait",
    "enhancement": "mlkit"
  }
}
```

### OCR Text Extraction

```bash
POST /ocr

# Option 1: Multipart file upload
curl -X POST http://localhost:5000/ocr \
  -F "image=@receipt.jpg"

# Option 2: JSON with file path
curl -X POST http://localhost:5000/ocr \
  -H "Content-Type: application/json" \
  -d '{"relativePath": "receipt.jpg"}'

Response:
{
  "text": "Extracted text from receipt...",
  "confidence": 87.5,
  "words": 125
}
```

## Processing Pipeline

### 1. Document Detection

The service tries multiple detection methods in order:

1. **Paper Color Detection**: Detects paper-colored regions (white/cream)
2. **GrabCut Center**: Assumes document is centered, uses GrabCut segmentation
3. **LSD Lines**: Line Segment Detector for edge-based detection
4. **Adaptive Threshold**: Thresholding-based edge detection
5. **DocTR Fallback**: Text-based document boundary detection

Each method is scored based on:
- Rectangularity (how rectangular the shape is)
- Area ratio (document size vs image size)
- Aspect ratio (plausible receipt dimensions)

### 2. Perspective Correction

If a quadrilateral is detected with high confidence:
- Compute perspective transform matrix
- Warp image to rectangular view
- Preserve aspect ratio

### 3. Image Enhancement

MLKit-style enhancement pipeline:
- **CLAHE**: Contrast Limited Adaptive Histogram Equalization
- **Denoising**: Bilateral filtering for noise reduction
- **Sharpening**: Unsharp mask for text clarity
- **Border Padding**: Add white border for better OCR

### 4. Orientation Correction

- Detect text orientation using DocTR or Tesseract OSD
- Rotate image to correct orientation (0°, 90°, 180°, 270°)

### 5. Border Trimming

- Remove uniform borders (solid color edges)
- Preserve document content
- Configurable tolerance and maximum trim

## Configuration

### Processing Config (`app/config.py`)

```python
class ProcessingConfig:
    # Detection
    MAX_DIMENSION_DETECTION = 1280
    MIN_DOCUMENT_AREA_RATIO = 0.20
    MAX_DOCUMENT_AREA_RATIO = 0.95
    METHOD_ORDER = ["paper_color", "grabcut_center", "lsd_lines", ...]

    # Scoring
    MIN_SCORE_FOR_WARP = 80
    MIN_SCORE_FOR_CROP = 40
    RECTANGULARITY_MIN_FOR_WARP = 0.88

    # Enhancement
    ENHANCE_PROFILE = 'mlkit'  # 'mlkit' | 'none'
    ENHANCE_MODE = 'color'     # 'color' | 'bw' | 'none'
    CLAHE_CLIP = 3.0
    CLAHE_TILE = 8

    # Output
    OUTPUT_FORMAT = 'webp'
    OUTPUT_QUALITY = 90
    OUTPUT_COLORSPACE = 'RGB'
```

## Docker Support

### Build Image

```bash
docker build -t tallylens-document-processor .
```

### Run Container

```bash
docker run -d \
  -p 5000:5000 \
  -v $(pwd)/uploads:/app/uploads \
  -e LOG_LEVEL=INFO \
  --name document-processor \
  tallylens-document-processor
```

### Docker Compose

```yaml
version: '3.8'
services:
  document-processor:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./uploads:/app/uploads
    environment:
      - LOG_LEVEL=INFO
      - OCR_LANG=eng+spa+nld
```

## Project Structure

```
document-processor/
├── app/
│   ├── __init__.py
│   ├── server.py                    # Flask application
│   ├── config.py                    # Configuration
│   ├── document_processor_hybrid.py # Main processing pipeline
│   ├── ocr_service.py               # OCR integration
│   ├── ml/
│   │   └── document_detector.py     # ML-based detection
│   └── utils/
│       ├── geometry.py              # Geometric utilities
│       ├── orientation.py           # Orientation detection
│       └── scoring.py               # Quality scoring
├── tests/
│   └── test_document_detection.py
├── uploads/                         # Upload directory
├── requirements.txt
├── Dockerfile
└── README.md
```

## Performance Tips

- **Low-resource mode**: Set `MAX_DIMENSION_DETECTION = 1280` to process smaller images
- **Disable ML detector**: Set `USE_ML_DETECTOR = false` if not needed
- **Caching**: Cache processed images to avoid reprocessing
- **Async processing**: Use background jobs for heavy processing
- **Batch processing**: Process multiple receipts in parallel

## Troubleshooting

### Tesseract Not Found

```bash
# Check installation
tesseract --version

# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Set path (if needed)
export TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata/
```

### OpenCV Import Error

```bash
# Install OpenCV dependencies
pip install opencv-python-headless

# For GUI support
pip install opencv-python
```

### Poor Detection Results

- Ensure good lighting and minimal shadows
- Use high-resolution images (1200px+ width)
- Avoid blurry or out-of-focus images
- Try different detection methods by adjusting `METHOD_ORDER`

## Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_document_detection.py

# With coverage
python -m pytest --cov=app tests/
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Write tests for new functionality
4. Ensure all tests pass
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

MIT

## Support

For issues and feature requests, please use the [GitHub Issues](https://github.com/mauro-mawesi/TALLYLENS_PYTHON/issues) page.

---

Built with ❤️ using Python and OpenCV
