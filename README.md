# AuthentiScan: AI-Generated Image Detection

> **Multi-Dataset Deepfake Detection with Cross-Validation & Explainability**

Production-ready deep learning system using EfficientNetB0 to detect AI-generated images across multiple domains with comprehensive evaluation and Grad-CAM visualization.

---

## 🎯 Project Overview

**AuthentiScan** (formerly TruthPixel) is a state-of-the-art deepfake detection system that trains on 3 large-scale datasets totaling **660K images from 9+ AI generators**. The system implements a novel 4-phase training approach with cross-dataset validation and Grad-CAM explainability.

### Key Innovations

1. **Cross-Dataset Validation**: Tests models on unseen datasets to measure true generalization
2. **Grad-CAM Explainability**: Visualizes what models look at when detecting fakes
3. **Multi-Model Ensemble**: Trains both dataset-specific and unified models
4. **Comprehensive Metrics**: Per-dataset, per-generator breakdown of performance

---

## 📊 Datasets

| Dataset | Images | Generators | Size | Purpose |
|---------|--------|------------|------|---------|
| **GenImage** | 400K | 8 (SD v1.4/1.5, Midjourney, GLIDE, ADM, VQDM, BigGAN, Wukong) | ~8GB | Multi-generator diversity |
| **CIFAKE** | 120K | Stable Diffusion v1.4 | ~3GB | General objects |
| **140K Faces** | 140K | StyleGAN | ~7GB | Deepfake detection |
| **Total** | **660K** | **9+ generators** | **~18GB** | Comprehensive training |

**Balanced Sampling Weights**:
- GenImage: 44% (400K images)
- CIFAKE: 18% (120K images)
- 140K Faces: 38% (140K images)

---

## 🏗️ Architecture

**EfficientNetB0-based Detector**:

```
Input (224x224x3)
    ↓
EfficientNet-B0 Backbone (pretrained ImageNet)
    ↓
GlobalAveragePooling2D
    ↓
Dense(512) + ReLU + Dropout(0.5)
    ↓
Dense(256) + ReLU + Dropout(0.3)
    ↓
Dense(1) + Sigmoid
```

- **Parameters**: ~5.3M total, ~1.5M trainable
- **Model Size**: ~22MB
- **Inference Time**: <150ms (GPU), ~1s (CPU)

---

## 🚀 4-Phase Training Strategy

### Phase 1: Baseline Models (12-18 hours)
Train separate models for each dataset:
- `genimage_baseline.pt`
- `cifake_baseline.pt`
- `faces_baseline.pt`

**Purpose**: Establish baseline performance per dataset

### Phase 2: Combined Model (18-24 hours)
Train unified model on all datasets with weighted sampling:
- Freeze backbone initially
- Unfreeze last 20 layers after epoch 5
- Checkpoint every 5 epochs

**Purpose**: Maximize generalization across all datasets

### Phase 3: Cross-Dataset Validation (2-3 hours)
Test each model on datasets it wasn't trained on:
- Measure generalization gap
- Identify failure modes
- Create performance heatmap

**Purpose**: Evaluate true generalization ability

### Phase 4: Grad-CAM Explainability (3-4 hours)
Generate activation heatmaps for all models:
- 20 samples per model/dataset combination
- Visualize model attention regions
- Compare model focus patterns

**Purpose**: Interpretability and trust

---

## ⚡ Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repo-url>
cd truthpixel

# Install dependencies
pip install -r requirements.txt

# Setup Kaggle API credentials
# 1. Go to https://www.kaggle.com/settings/account
# 2. Create New API Token
# 3. Move kaggle.json to ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 2. Download Datasets (~18GB)

```bash
# Automated download (CIFAKE + Faces) + Manual instructions (GenImage)
bash scripts/download_datasets.sh

# Manual GenImage download:
# Visit: https://drive.google.com/drive/folders/1jGt10bwTbhEZuGXLyvrCuxOI0cBqQ1FS
# Download all folders to data/downloads/genimage/
```

### 3. Organize Datasets

```bash
# Organize all datasets into train/val/test splits
python src/organize_datasets.py --datasets genimage cifake faces

# Verify organization
python scripts/verify_datasets.py
```

### 4. Train Models

```bash
# Option A: Full pipeline (~48-55 hours)
bash scripts/train_all.sh

# Option B: Step-by-step
python src/train_baseline.py --dataset genimage  # Phase 1
python src/train_baseline.py --dataset cifake
python src/train_baseline.py --dataset faces

python src/train_combined.py                     # Phase 2

python src/cross_validate.py                     # Phase 3

python src/gradcam.py --model all --samples 20   # Phase 4
```

### 5. Evaluate

```bash
# Comprehensive evaluation
python src/evaluate.py

# View training logs
tensorboard --logdir results/logs
```

### 6. Launch Web App

```bash
streamlit run app/streamlit_app.py
```

---

## 📈 Expected Results

### Training Time (RTX 4050, 6GB VRAM)
- Phase 1 (Baselines): 12-18 hours
- Phase 2 (Combined): 18-24 hours
- Phase 3 (Cross-val): 2-3 hours
- Phase 4 (Grad-CAM): 3-4 hours
- **Total: 40-50 hours**

### Performance Targets
- **Baseline Models**: 90-95% accuracy on respective datasets
- **Combined Model**: 92-96% accuracy overall
- **Cross-Dataset Generalization**: 85-92%
- **Generalization Gap**: <8%

---

## 📁 Project Structure

```
truthpixel/
├── config.yaml                 # Configuration file
├── requirements.txt            # Dependencies
│
├── data/                       # Datasets (18GB total)
│   ├── genimage/              # 400K images, 8 generators
│   ├── cifake/                # 120K images
│   └── faces/                 # 140K images
│
├── src/                        # Source code
│   ├── train_baseline.py      # Phase 1: Baseline training
│   ├── train_combined.py      # Phase 2: Combined model
│   ├── cross_validate.py      # Phase 3: Cross-validation
│   ├── gradcam.py             # Phase 4: Grad-CAM
│   ├── evaluate.py            # Comprehensive evaluation
│   ├── data_loader_multi.py   # Multi-dataset loader
│   ├── augmentation.py        # Data augmentation
│   ├── download_multi_datasets.py  # Dataset download
│   └── organize_datasets.py   # Dataset organization
│
├── scripts/                    # Helper scripts
│   ├── download_datasets.sh   # Automated download
│   ├── verify_datasets.py     # Dataset verification
│   └── train_all.sh           # Full training pipeline
│
├── app/                        # Streamlit web interface
│   └── streamlit_app.py       # Web app
│
├── models/                     # Trained models
│   ├── baseline/              # Dataset-specific models
│   │   ├── genimage_baseline.pt
│   │   ├── cifake_baseline.pt
│   │   └── faces_baseline.pt
│   ├── combined/              # Unified model
│   │   └── combined_model.pt
│   └── checkpoints/           # Training checkpoints
│
└── results/                    # Evaluation results
    ├── metrics/               # JSON metrics
    ├── plots/                 # Visualizations
    ├── cross_validation/      # Cross-val results
    ├── gradcam/               # Grad-CAM heatmaps
    └── logs/                  # TensorBoard logs
```

---

## 🛠️ Hardware Requirements

**Minimum**:
- GPU: 6GB VRAM (RTX 3060, RTX 4050)
- RAM: 16GB
- Storage: 30GB (datasets + models + results)

**Recommended**:
- GPU: 8GB+ VRAM (RTX 3070, RTX 4060)
- RAM: 32GB
- Storage: 50GB SSD

**Settings for 6GB VRAM**:
- Batch size: 16 (adjust in `config.yaml`)
- Mixed precision training (FP16)
- Gradient checkpointing enabled

---

## 📚 Usage Examples

### Training Individual Models

```bash
# Train GenImage baseline
python src/train_baseline.py --dataset genimage --config config.yaml

# Train with custom config
python src/train_baseline.py --dataset cifake --config custom_config.yaml
```

### Evaluation

```bash
# Evaluate all models
python src/evaluate.py

# Cross-validation only
python src/cross_validate.py

# Grad-CAM for specific model
python src/gradcam.py --model genimage --samples 50
```

### Inference (via Streamlit)

1. Launch app: `streamlit run app/streamlit_app.py`
2. Upload image
3. Select model(s)
4. View prediction + Grad-CAM heatmap

---

## 🔬 Novel Contributions

1. **Cross-Dataset Validation Framework**: Systematic evaluation of generalization across datasets
2. **4-Phase Progressive Training**: Baseline → Combined → Cross-Val → Explainability
3. **Multi-Generator Coverage**: 9+ AI generators (SD, Midjourney, GLIDE, StyleGAN, etc.)
4. **Grad-CAM Integration**: Built-in explainability for all models
5. **Comprehensive Benchmarking**: Per-dataset, per-generator performance breakdown

---

## 📊 Evaluation Metrics

The system tracks:
- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1**: Class-specific performance
- **AUC-ROC**: Discrimination ability
- **Confusion Matrix**: Error analysis
- **Per-Dataset Breakdown**: Performance on each dataset
- **Per-Generator Breakdown**: Performance on each AI generator
- **Generalization Gap**: Same-dataset vs cross-dataset accuracy

---

## 🐛 Troubleshooting

### Out of Memory (OOM)
- Reduce batch size in `config.yaml` (16 → 8)
- Enable mixed precision training
- Reduce number of workers

### Slow Training
- Increase batch size if VRAM available
- Enable pin_memory for faster GPU transfer
- Use SSD for dataset storage

### Dataset Download Issues
- CIFAKE/Faces: Verify Kaggle API credentials
- GenImage: Manual download required (Google Drive)

---

## 📖 Documentation

- **QUICKSTART.md**: Step-by-step beginner guide
- **docs/DATASET_GUIDE.md**: Dataset details and download
- **docs/TRAINING_GUIDE.md**: Training methodology
- **docs/EVALUATION_GUIDE.md**: Metrics and analysis

---

## 🙏 Acknowledgments

**Datasets**:
- GenImage: [GenImage Benchmark](https://github.com/GenImage/GenImage)
- CIFAKE: [Kaggle - CIFAKE Dataset](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
- 140K Faces: [Kaggle - 140K Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)

**Architecture**:
- EfficientNet: [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- PyTorch: Deep learning framework

---

## 📄 License

This project is for educational and research purposes.

---

## 📧 Contact

For questions or issues, please open a GitHub issue.

---

**Built with ❤️ for robust AI-generated image detection**
