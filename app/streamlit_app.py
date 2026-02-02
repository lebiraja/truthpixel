"""
Streamlit Web Application for TruthPixel AI-Generated Image Detection.

This app provides a user-friendly interface for:
- Uploading images
- Real-time prediction (Real vs AI-Generated)
- Confidence score display
- Grad-CAM visualization
"""

import os
import sys
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from gradcam import GradCAM


# Page configuration
st.set_page_config(
    page_title="TruthPixel - AI Image Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS for styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .prediction-real {
        font-size: 2rem;
        font-weight: bold;
        color: #2ECC71;
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        background-color: #E8F8F5;
    }
    .prediction-ai {
        font-size: 2rem;
        font-weight: bold;
        color: #E74C3C;
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        background-color: #FADBD8;
    }
    .confidence-score {
        font-size: 1.5rem;
        text-align: center;
        margin-top: 1rem;
    }
    .info-box {
        padding: 1rem;
        border-radius: 10px;
        background-color: #F0F2F6;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_path: str = '../models/truthpixel_final.h5'):
    """
    Load the trained model (cached for performance).

    Args:
        model_path: Path to the trained model

    Returns:
        Loaded Keras model
    """
    try:
        # Try relative path first
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
        # Try absolute path
        elif os.path.exists(os.path.join(os.path.dirname(__file__), model_path)):
            model_path = os.path.join(os.path.dirname(__file__), model_path)
            model = tf.keras.models.load_model(model_path)
        # Try from root
        else:
            model_path = 'models/truthpixel_final.h5'
            model = tf.keras.models.load_model(model_path)

        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please ensure the model file exists at: models/truthpixel_final.h5")
        return None


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess uploaded image for model input.

    Args:
        image: PIL Image

    Returns:
        Preprocessed image array
    """
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Resize to model input size
    image = image.resize((224, 224))

    # Convert to array and normalize
    img_array = np.array(image) / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def predict_image(model, img_array: np.ndarray):
    """
    Generate prediction for an image.

    Args:
        model: Trained Keras model
        img_array: Preprocessed image array

    Returns:
        Tuple of (prediction_label, confidence)
    """
    # Get prediction
    prediction = model.predict(img_array, verbose=0)[0][0]

    # Determine class
    if prediction > 0.5:
        label = "AI-Generated"
        confidence = float(prediction)
    else:
        label = "Real"
        confidence = float(1 - prediction)

    return label, confidence


def generate_gradcam(model, img_array: np.ndarray, original_img: np.ndarray):
    """
    Generate Grad-CAM heatmap for the image.

    Args:
        model: Trained model
        img_array: Preprocessed image
        original_img: Original image for overlay

    Returns:
        Overlayed image with heatmap
    """
    try:
        # Create Grad-CAM instance
        gradcam = GradCAM(model=model)

        # Generate heatmap
        heatmap = gradcam.get_gradcam_heatmap(img_array)

        # Overlay on original image
        overlayed = gradcam.overlay_heatmap_on_image(original_img, heatmap)

        return heatmap, overlayed
    except Exception as e:
        st.warning(f"Could not generate Grad-CAM: {e}")
        return None, None


def main():
    """
    Main Streamlit application.
    """
    # Header
    st.markdown('<div class="main-header">🔍 TruthPixel</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">AI-Generated Image Detection with Deep Learning</div>',
        unsafe_allow_html=True
    )

    # Sidebar
    with st.sidebar:
        st.header("📊 About")
        st.info(
            """
            **TruthPixel** uses a deep learning model based on EfficientNetB0
            to detect whether an image is real or AI-generated.

            The model was trained on the CIFAKE dataset containing 120,000 images.
            """
        )

        st.header("📈 Model Performance")
        st.metric("Accuracy", ">92%")
        st.metric("F1-Score", ">0.92")
        st.metric("Dataset", "CIFAKE (120K images)")

        st.header("🔬 Features")
        st.markdown("""
        - Real-time image classification
        - Confidence score display
        - Grad-CAM explainability
        - Support for JPG, JPEG, PNG
        """)

        st.header("ℹ️ How to Use")
        st.markdown("""
        1. Upload an image using the file uploader
        2. Click "Analyze Image"
        3. View the prediction and confidence
        4. Explore the Grad-CAM heatmap
        """)

    # Main content
    st.markdown("---")

    # Load model
    model = load_model()

    if model is None:
        st.error("Failed to load model. Please check the model file path.")
        return

    st.success("✓ Model loaded successfully!")

    # File uploader
    st.header("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image file (JPG, JPEG, PNG)",
        type=['jpg', 'jpeg', 'png'],
        help="Upload an image to detect if it's real or AI-generated"
    )

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📷 Uploaded Image")
            st.image(image, use_container_width=True)

        # Analyze button
        if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
            with st.spinner("Analyzing image..."):
                # Preprocess image
                img_array = preprocess_image(image)
                original_img = np.array(image.resize((224, 224))) / 255.0

                # Get prediction
                label, confidence = predict_image(model, img_array)

                # Display prediction
                st.markdown("---")
                st.header("📊 Prediction Results")

                if label == "Real":
                    st.markdown(
                        f'<div class="prediction-real">✓ {label} Image</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="prediction-ai">⚠ {label} Image</div>',
                        unsafe_allow_html=True
                    )

                # Confidence score
                st.markdown(
                    f'<div class="confidence-score">Confidence: {confidence:.2%}</div>',
                    unsafe_allow_html=True
                )

                # Progress bar for confidence
                st.progress(confidence)

                # Grad-CAM visualization
                st.markdown("---")
                st.header("🔬 Grad-CAM Explainability")

                st.info(
                    "The heatmap below shows which parts of the image influenced the model's decision. "
                    "Red/yellow regions had the most impact on the prediction."
                )

                with st.spinner("Generating Grad-CAM visualization..."):
                    heatmap, overlayed = generate_gradcam(model, img_array, original_img)

                    if heatmap is not None and overlayed is not None:
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.subheader("Original")
                            st.image(original_img, use_container_width=True)

                        with col2:
                            st.subheader("Heatmap")
                            fig, ax = plt.subplots()
                            ax.imshow(heatmap, cmap='jet')
                            ax.axis('off')
                            st.pyplot(fig)
                            plt.close()

                        with col3:
                            st.subheader("Overlay")
                            st.image(overlayed, use_container_width=True)

                        st.success("✓ Analysis complete!")
                    else:
                        st.warning("Could not generate Grad-CAM visualization.")

        # Additional information
        with col2:
            st.subheader("📝 Image Information")
            st.markdown(f"""
            <div class="info-box">
            <b>Filename:</b> {uploaded_file.name}<br>
            <b>Size:</b> {image.size[0]} x {image.size[1]} pixels<br>
            <b>Format:</b> {image.format}<br>
            <b>Mode:</b> {image.mode}
            </div>
            """, unsafe_allow_html=True)

    else:
        # Show example instructions
        st.info("👆 Please upload an image to get started")

        st.markdown("---")
        st.header("📚 Example Use Cases")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 🖼️ Verify Authenticity")
            st.write("Check if an image is real or AI-generated for content moderation.")

        with col2:
            st.markdown("### 🎨 Detect AI Art")
            st.write("Identify AI-generated artwork in your digital collections.")

        with col3:
            st.markdown("### 📰 Combat Misinformation")
            st.write("Verify images in news and social media to prevent fake content spread.")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; padding: 2rem;">
        <p>Built with ❤️ using TensorFlow, EfficientNetB0, and Streamlit</p>
        <p>Trained on CIFAKE dataset | Model achieves >92% accuracy</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
