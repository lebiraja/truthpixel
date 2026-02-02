"""
TruthPixel Web App
Flask server for AI image detection
"""

from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from inference import TruthPixelDetector

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model once at startup
print("Loading TruthPixel model...")
detector = TruthPixelDetector(model_path='../models/phase_2_best.pth')
print("✓ Model loaded and ready!")


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle image upload and prediction
    """
    # Check if file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    # Check if filename is empty
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Check if file type is allowed
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, GIF, BMP, WEBP'}), 400

    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Make prediction
        result = detector.predict(filepath)

        # Clean up uploaded file
        os.remove(filepath)

        # Return results
        return jsonify({
            'success': True,
            'prediction': result['prediction'],
            'confidence': round(result['confidence'], 2),
            'is_fake': result['is_fake']
        })

    except Exception as e:
        # Clean up on error
        if os.path.exists(filepath):
            os.remove(filepath)

        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model': 'loaded'})


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🚀 TruthPixel Web App Starting...")
    print("=" * 60)
    print("📍 URL: http://localhost:5000")
    print("🧠 Model: Loaded and ready")
    print("🎯 Accuracy: 98.41%")
    print("=" * 60 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)
