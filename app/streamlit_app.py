"""
AuthentiScan Web Interface
==========================

Streamlit application for AI-Generated Image Detection.
Features:
- Multi-model support (GenImage, CIFAKE, Faces, Combined)
- Real-time prediction with confidence scores
- Grad-CAM explainability visualization
- Batch processing
- Model comparison
- Cross-validation results

Usage:
    streamlit run app/streamlit_app.py
"""

import streamlit as st
import torch
import torch.nn as nn
import timm
import numpy as np
from PIL import Image
import sys
from pathlib import Path
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import cv2

# Add src to path
SRC_PATH = Path(__file__).parent.parent / 'src'
sys.path.append(str(SRC_PATH))

from augmentation import get_validation_transforms

# Page config
st.set_page_config(
    page_title="AuthentiScan - AI Image Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-card {
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .real {
        background-color: #E8F5E9;
        border-left: 5px solid #4CAF50;
    }
    .fake {
        background-color: #FFEBEE;
        border-left: 5px solid #F44336;
    }
    .uncertain {
        background-color: #FFF8E1;
        border-left: 5px solid #FFC107;
    }
    .metric-card {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# Model Definition (Matching training scripts)
# ==============================================================================
class DeepfakeDetector(nn.Module):
    """EfficientNetB0-based deepfake detector."""

    def __init__(self, pretrained=False, dropout=0.5):
        super(DeepfakeDetector, self).__init__()

        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=pretrained,
            num_classes=0,
            global_pool=''
        )

        # Get feature dimension (1280 for EfficientNet-B0)
        feature_dim = 1280

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.backbone(x)
        features = self.pool(features)
        features = self.flatten(features)
        output = self.classifier(features)
        return output


# ==============================================================================
# Helper Functions
# ==============================================================================
@st.cache_resource
def load_model(model_name, device):
    """Load a specific model from checkpoint."""
    model_path = None
    base_path = Path("models")

    if model_name == "Combined Model":
        model_path = base_path / "combined" / "combined_model_best.pt"
    else:
        # Extract dataset name from "X Baseline"
        dataset = model_name.split()[0].lower()
        model_path = base_path / "baseline" / f"{dataset}_baseline_best.pt"

    if not model_path.exists():
        # Try finding standard checkpoint if best doesn't exist
        fallback_path = str(model_path).replace("_best.pt", ".pt")
        if Path(fallback_path).exists():
            model_path = Path(fallback_path)
        else:
            return None, f"Model file not found: {model_path}"

    try:
        model = DeepfakeDetector(pretrained=False)
        checkpoint = torch.load(model_path, map_location=device)

        # Handle state dict structure
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model, None
    except Exception as e:
        return None, str(e)


def get_gradcam(model, input_tensor, device):
    """Generate Grad-CAM heatmap."""
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

    target_layer = model.backbone.blocks[-1]
    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=(device.type == 'cuda'))

    # Target: 1 for Fake, 0 for Real. We want to see what triggers the prediction.
    # Get prediction first
    with torch.no_grad():
        output = model(input_tensor)
        pred = output.item()
        target_class = 1 if pred > 0.5 else 0

    targets = [BinaryClassifierOutputTarget(target_class)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    return grayscale_cam[0, :]


def predict(model, image, device):
    """Run prediction on a single image."""
    # Preprocess
    transform = get_validation_transforms(img_size=224)
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        prob = output.item()

    return prob, img_tensor


# ==============================================================================
# Main App
# ==============================================================================
def main():
    # Header
    st.markdown('<h1 class="main-header">🛡️ AuthentiScan</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Advanced AI-Generated Image Detection System</p>',
        unsafe_allow_html=True
    )

    # Sidebar
    st.sidebar.title("Configuration")

    # Navigation
    app_mode = st.sidebar.selectbox(
        "Navigate",
        ["Single Analysis", "Batch Processing", "Model Comparison", "Cross-Validation", "About"]
    )

    # Hardware
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    st.sidebar.info(f"Running on: **{device_name.upper()}**")

    # Models
    available_models = ["Combined Model", "GenImage Baseline", "CIFAKE Baseline", "Faces Baseline"]

    # ==========================================================================
    # Single Analysis Mode
    # ==========================================================================
    if app_mode == "Single Analysis":
        st.header("🔍 Single Image Analysis")

        selected_model_name = st.sidebar.selectbox("Select Model", available_models)

        # Load model
        with st.spinner(f"Loading {selected_model_name}..."):
            model, error = load_model(selected_model_name, device)

        if error:
            st.error(f"Failed to load model: {error}")
            st.warning("Have you trained the models yet? Run `bash scripts/train_all.sh` first.")
            return

        uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png', 'webp'])

        if uploaded_file:
            col1, col2 = st.columns([1, 1])

            # Display Image
            image = Image.open(uploaded_file).convert('RGB')
            with col1:
                st.subheader("Input Image")
                st.image(image, use_column_width=True)

            # Prediction
            with col2:
                st.subheader("Analysis")
                with st.spinner("Analyzing..."):
                    prob, img_tensor = predict(model, image, device)

                    # Grad-CAM
                    try:
                        heatmap = get_gradcam(model, img_tensor, device)

                        # Overlay
                        from pytorch_grad_cam.utils.image import show_cam_on_image
                        img_np = np.array(image.resize((224, 224))) / 255.0
                        vis = show_cam_on_image(img_np, heatmap, use_rgb=True)

                        show_gradcam = st.checkbox("Show Explanation (Grad-CAM)", value=True)
                        if show_gradcam:
                            st.image(vis, caption="Model Attention Map", use_column_width=True)

                    except Exception as e:
                        st.warning(f"Could not generate Grad-CAM: {e}")

            # Results Display
            is_fake = prob > 0.5
            confidence = prob if is_fake else 1 - prob
            label = "AI-Generated" if is_fake else "Real"
            css_class = "fake" if is_fake else "real"

            if 0.4 < prob < 0.6:
                css_class = "uncertain"
                label = "Uncertain"
                st.warning("⚠️ Model is uncertain about this image.")

            st.markdown(f"""
            <div class="result-card {css_class}">
                <h2 style="margin:0; text-align:center;">{label}</h2>
                <h3 style="margin:0; text-align:center;">Confidence: {confidence:.2%}</h3>
            </div>
            """, unsafe_allow_html=True)

            # Detailed Metrics
            m1, m2 = st.columns(2)
            m1.metric("Fake Probability", f"{prob:.2%}")
            m2.metric("Real Probability", f"{1-prob:.2%}")

    # ==========================================================================
    # Batch Processing Mode
    # ==========================================================================
    elif app_mode == "Batch Processing":
        st.header("📂 Batch Processing")

        selected_model_name = st.sidebar.selectbox("Select Model", available_models)
        model, error = load_model(selected_model_name, device)

        if error:
            st.error(error)
            return

        uploaded_files = st.file_uploader(
            "Upload Images",
            type=['jpg', 'jpeg', 'png', 'webp'],
            accept_multiple_files=True
        )

        if uploaded_files:
            if st.button(f"Analyze {len(uploaded_files)} Images"):
                results = []
                progress_bar = st.progress(0)

                for idx, file in enumerate(uploaded_files):
                    img = Image.open(file).convert('RGB')
                    prob, _ = predict(model, img, device)

                    results.append({
                        "Filename": file.name,
                        "Prediction": "AI-Generated" if prob > 0.5 else "Real",
                        "Confidence": prob if prob > 0.5 else 1 - prob,
                        "Fake_Prob": prob
                    })
                    progress_bar.progress((idx + 1) / len(uploaded_files))

                df = pd.DataFrame(results)

                # Summary
                st.success("Analysis Complete!")
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Images", len(df))
                c2.metric("Detected Fakes", len(df[df["Prediction"] == "AI-Generated"]))
                c3.metric("Detected Real", len(df[df["Prediction"] == "Real"]))

                # Table
                st.dataframe(df.style.format({"Confidence": "{:.2%}", "Fake_Prob": "{:.2%}"}))

                # Download
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download CSV",
                    csv,
                    "authentiscan_results.csv",
                    "text/csv",
                    key='download-csv'
                )

    # ==========================================================================
    # Model Comparison Mode
    # ==========================================================================
    elif app_mode == "Model Comparison":
        st.header("⚖️ Model Comparison")
        st.info("Compare predictions from all models on the same image.")

        uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png', 'webp'])

        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, width=300, caption="Input Image")

            if st.button("Run All Models"):
                cols = st.columns(len(available_models))

                for idx, model_name in enumerate(available_models):
                    with cols[idx]:
                        st.markdown(f"**{model_name}**")
                        model, error = load_model(model_name, device)

                        if error:
                            st.warning("Model not available")
                            continue

                        with st.spinner("Running..."):
                            prob, img_tensor = predict(model, image, device)
                            heatmap = get_gradcam(model, img_tensor, device)

                            is_fake = prob > 0.5
                            label = "FAKE" if is_fake else "REAL"
                            color = "red" if is_fake else "green"

                            st.markdown(f"<h3 style='color:{color}'>{label}</h3>", unsafe_allow_html=True)
                            st.write(f"Conf: {prob if is_fake else 1-prob:.1%}")

                            # Grad-CAM
                            from pytorch_grad_cam.utils.image import show_cam_on_image
                            img_np = np.array(image.resize((224, 224))) / 255.0
                            vis = show_cam_on_image(img_np, heatmap, use_rgb=True)
                            st.image(vis, caption="Attention")

    # ==========================================================================
    # Cross-Validation Results
    # ==========================================================================
    elif app_mode == "Cross-Validation":
        st.header("🔄 Cross-Dataset Generalization")

        # Load results if available
        cv_path = Path("results/cross_validation/cross_validation_results.csv")
        heatmap_path = Path("results/cross_validation/cross_validation_heatmap.png")

        if heatmap_path.exists():
            st.image(str(heatmap_path), caption="Generalization Matrix", use_column_width=True)

        if cv_path.exists():
            df = pd.read_csv(cv_path)
            st.dataframe(df)
        else:
            st.info("No cross-validation results found. Run `python src/cross_validate.py` first.")

        st.markdown("""
        ### Understanding the Results
        - **Diagonal**: Performance on training domain (expected to be high).
        - **Off-Diagonal**: Performance on unseen domains (generalization).
        - **Combined Model**: Should show consistent performance across all datasets.
        """)

    # ==========================================================================
    # About
    # ==========================================================================
    elif app_mode == "About":
        st.header("About AuthentiScan")
        st.markdown("""
        **AuthentiScan** is a production-grade AI detection system trained on 660K images.

        ### Datasets
        - **GenImage**: 400K images (8 generators: SD, Midjourney, etc.)
        - **CIFAKE**: 120K images (Stable Diffusion)
        - **Faces**: 140K images (StyleGAN)

        ### Architecture
        - **Backbone**: EfficientNet-B0
        - **Training**: 4-Phase Progressive Strategy
        - **Explainability**: Grad-CAM Integration

        ### Team
        Developed by the TruthPixel Team.
        """)

if __name__ == "__main__":
    main()
