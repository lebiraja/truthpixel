// TruthPixel Web App JavaScript

let selectedFile = null;

// DOM Elements
const uploadBox = document.getElementById('uploadBox');
const fileInput = document.getElementById('fileInput');
const previewSection = document.getElementById('previewSection');
const previewImage = document.getElementById('previewImage');
const removeBtn = document.getElementById('removeBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const loading = document.getElementById('loading');
const results = document.getElementById('results');
const error = document.getElementById('error');
const errorText = document.getElementById('errorText');
const resultCard = document.getElementById('resultCard');
const resultIcon = document.getElementById('resultIcon');
const resultText = document.getElementById('resultText');
const confidenceFill = document.getElementById('confidenceFill');
const confidenceText = document.getElementById('confidenceText');
const newBtn = document.getElementById('newBtn');

// Click to upload
uploadBox.addEventListener('click', () => {
    fileInput.click();
});

// File selection
fileInput.addEventListener('change', (e) => {
    handleFile(e.target.files[0]);
});

// Drag and drop
uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.classList.add('dragging');
});

uploadBox.addEventListener('dragleave', () => {
    uploadBox.classList.remove('dragging');
});

uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.classList.remove('dragging');
    handleFile(e.dataTransfer.files[0]);
});

// Handle file selection
function handleFile(file) {
    if (!file) return;

    // Validate file type
    const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/bmp', 'image/webp'];
    if (!validTypes.includes(file.type)) {
        showError('Invalid file type. Please upload an image (PNG, JPG, JPEG, GIF, BMP, WEBP)');
        return;
    }

    // Validate file size (16MB max)
    if (file.size > 16 * 1024 * 1024) {
        showError('File too large. Maximum size is 16MB');
        return;
    }

    selectedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        uploadBox.style.display = 'none';
        previewSection.style.display = 'block';
        analyzeBtn.style.display = 'block';
        hideError();
        hideResults();
    };
    reader.readAsDataURL(file);
}

// Remove image
removeBtn.addEventListener('click', () => {
    resetUpload();
});

// Analyze image
analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    // Show loading
    analyzeBtn.style.display = 'none';
    loading.style.display = 'block';
    hideError();
    hideResults();

    // Create form data
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        // Send to server
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        // Hide loading
        loading.style.display = 'none';

        if (data.success) {
            showResults(data);
        } else {
            showError(data.error || 'Prediction failed');
            analyzeBtn.style.display = 'block';
        }

    } catch (err) {
        loading.style.display = 'none';
        showError('Network error. Please try again.');
        analyzeBtn.style.display = 'block';
    }
});

// Show results
function showResults(data) {
    // Set card style based on prediction
    if (data.is_fake) {
        resultCard.className = 'result-card fake';
        resultIcon.textContent = '🤖';
        resultText.textContent = 'AI-Generated Image';
    } else {
        resultCard.className = 'result-card real';
        resultIcon.textContent = '✓';
        resultText.textContent = 'Real Image';
    }

    // Animate confidence bar
    setTimeout(() => {
        confidenceFill.style.width = data.confidence + '%';
    }, 100);

    confidenceText.textContent = `Confidence: ${data.confidence}%`;

    // Show results
    results.style.display = 'block';
}

// Show error
function showError(message) {
    errorText.textContent = message;
    error.style.display = 'block';
}

// Hide error
function hideError() {
    error.style.display = 'none';
}

// Hide results
function hideResults() {
    results.style.display = 'none';
    confidenceFill.style.width = '0%';
}

// New analysis
newBtn.addEventListener('click', () => {
    resetUpload();
});

// Reset upload
function resetUpload() {
    selectedFile = null;
    fileInput.value = '';
    uploadBox.style.display = 'block';
    previewSection.style.display = 'none';
    analyzeBtn.style.display = 'none';
    hideError();
    hideResults();
}

// Prevent default drag behavior on document
document.addEventListener('dragover', (e) => e.preventDefault());
document.addEventListener('drop', (e) => e.preventDefault());
