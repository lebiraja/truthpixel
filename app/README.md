# 🔍 TruthPixel Web App

Beautiful web interface for AI-generated image detection.

## Features

✅ **Drag & Drop Upload** - Easy image uploading
✅ **Real-time Prediction** - Instant AI detection
✅ **98.41% Accuracy** - Powered by trained PyTorch model
✅ **Beautiful UI** - Modern gradient design
✅ **Responsive** - Works on mobile and desktop

## Quick Start

### 1. Start the Server

```bash
cd app
python3 app.py
```

### 2. Open Browser

Visit: **http://localhost:5000**

### 3. Upload an Image

- Drag & drop an image
- Or click to browse
- Click "Analyze Image"
- Get instant results!

## API Endpoints

### `GET /`
Main web interface

### `POST /predict`
Upload image for prediction

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: `file` (image file)

**Response:**
```json
{
    "success": true,
    "prediction": "AI-Generated (FAKE)",
    "confidence": 98.5,
    "is_fake": true
}
```

### `GET /health`
Health check endpoint

**Response:**
```json
{
    "status": "healthy",
    "model": "loaded"
}
```

## Supported Image Formats

- PNG
- JPG/JPEG
- GIF
- BMP
- WEBP

**Max file size:** 16MB

## How It Works

1. **Upload** - User uploads an image via drag-drop or file picker
2. **Preprocess** - Image resized to 224x224, ImageNet normalized
3. **Predict** - EfficientNet-B0 model makes prediction
4. **Display** - Results shown with confidence score

## Technology Stack

- **Backend:** Flask (Python)
- **Frontend:** HTML5, CSS3, Vanilla JavaScript
- **ML Model:** PyTorch + EfficientNet-B0
- **Accuracy:** 98.41% validation accuracy

## Project Structure

```
app/
├── app.py              # Flask server
├── inference.py        # Model loading & prediction
├── templates/
│   └── index.html     # Frontend HTML
├── static/
│   ├── style.css      # Styling
│   └── script.js      # JavaScript logic
└── uploads/           # Temporary upload folder
```

## Performance

- Model load time: ~2 seconds
- Prediction time: ~50ms (GPU) / ~200ms (CPU)
- Supports: CUDA GPU acceleration

## Security

- File type validation
- File size limits (16MB)
- Secure filename handling
- Automatic cleanup of uploaded files

## Troubleshooting

**Model not loading?**
```bash
# Make sure model file exists
ls ../models/phase_2_best.pth
```

**Port already in use?**
Change port in `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)
```

**CUDA out of memory?**
Model will automatically fall back to CPU

## Credits

- Model: EfficientNet-B0 (torchvision)
- Dataset: CIFAKE (79,203 images)
- Training: PyTorch transfer learning
