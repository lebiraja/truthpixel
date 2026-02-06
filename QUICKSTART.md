# 🚀 AuthentiScan Quick Start Guide

Get started with AuthentiScan in minutes! This guide walks you through the complete setup and training process.

---

## Prerequisites

- Python 3.8+
- NVIDIA GPU with 6GB+ VRAM (RTX 3060, RTX 4050 or better)
- 30GB free disk space
- Kaggle account (for dataset download)

---

## Step 1: Setup Environment

```bash
# Clone the repository
git clone <repo-url>
cd truthpixel

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Step 2: Setup Kaggle API

CIFAKE and Faces datasets require Kaggle API credentials:

```bash
# 1. Go to https://www.kaggle.com/settings/account
# 2. Scroll to "API" section
# 3. Click "Create New Token"
# 4. This downloads kaggle.json

# 5. Move kaggle.json to the correct location
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# 6. Verify setup
kaggle --version
```

---

## Step 3: Download Datasets (~18GB)

```bash
bash scripts/download_datasets.sh
```

**Manual GenImage Download** (required):
1. Visit: https://drive.google.com/drive/folders/1jGt10bwTbhEZuGXLyvrCuxOI0cBqQ1FS
2. Download all folders to `data/downloads/genimage/`

---

## Step 4: Organize Datasets

```bash
python src/organize_datasets.py --datasets genimage cifake faces
python scripts/verify_datasets.py  # Verify
```

---

## Step 5: Train Models

### Full Pipeline (~48-55 hours)
```bash
bash scripts/train_all.sh
```

### Or Step-by-Step
```bash
# Phase 1: Baseline models (12-18 hours)
python src/train_baseline.py --dataset genimage
python src/train_baseline.py --dataset cifake
python src/train_baseline.py --dataset faces

# Phase 2: Combined model (18-24 hours)
python src/train_combined.py

# Phase 3: Cross-validation (2-3 hours)
python src/cross_validate.py

# Phase 4: Grad-CAM (3-4 hours)
python src/gradcam.py --model all --samples 20
```

---

## Step 6: Evaluate & Visualize

```bash
# Comprehensive evaluation
python src/evaluate.py

# View training logs
tensorboard --logdir results/logs

# Launch web app
streamlit run app/streamlit_app.py
```

---

## Common Commands

```bash
# Dataset management
bash scripts/download_datasets.sh
python src/organize_datasets.py --datasets all
python scripts/verify_datasets.py

# Training
bash scripts/train_all.sh
python src/train_baseline.py --dataset genimage

# Evaluation
python src/evaluate.py
tensorboard --logdir results/logs
streamlit run app/streamlit_app.py
```

---

## Troubleshooting

**Out of Memory**: Reduce `batch_size` in `config.yaml` (16 → 8)

**Slow Training**: Use SSD, increase batch size if VRAM available

**Dataset Issues**: Verify Kaggle credentials, manually download GenImage

---

For detailed guides, see **README.md** and **docs/** directory.

**Ready to start? Begin with Step 1! 🚀**
