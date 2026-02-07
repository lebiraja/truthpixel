#!/bin/bash

# AuthentiScan Complete Training Pipeline
# Runs all 4 phases of training

set -e  # Exit on error

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "=========================================================================="
echo "  AuthentiScan Complete Training Pipeline"
echo "=========================================================================="
echo "Total estimated time: 48-55 hours on RTX 4050"
echo ""
echo "4 Phases:"
echo "  1. Baseline models (3 models, ~12-18 hours)"
echo "  2. Combined model (~18-24 hours)"
echo "  3. Cross-validation (~2-3 hours)"
echo "  4. Grad-CAM explainability (~3-4 hours)"
echo ""
read -p "Press Enter to start training, or Ctrl+C to cancel..."

# Create necessary directories
mkdir -p models/{checkpoints,baseline,combined}
mkdir -p results/{logs,metrics,plots,cross_validation,gradcam}

# Phase 1: Baseline models
echo ""
echo "=========================================================================="
echo -e "${BLUE}PHASE 1: Training baseline models (12-18 hours)${NC}"
echo "=========================================================================="
echo ""

echo -e "${GREEN}Training GenImage baseline model...${NC}"
python src/train_baseline.py --dataset genimage

echo ""
echo -e "${GREEN}Training CIFAKE baseline model...${NC}"
python src/train_baseline.py --dataset cifake

echo ""
echo -e "${GREEN}Training Faces baseline model...${NC}"
python src/train_baseline.py --dataset faces

echo ""
echo -e "${GREEN}✓ Phase 1 complete: All baseline models trained${NC}"

# Phase 2: Combined model
echo ""
echo "=========================================================================="
echo -e "${BLUE}PHASE 2: Training combined model (18-24 hours)${NC}"
echo "=========================================================================="
echo ""

python src/train_combined.py

echo ""
echo -e "${GREEN}✓ Phase 2 complete: Combined model trained${NC}"

# Phase 3: Cross-validation
echo ""
echo "=========================================================================="
echo -e "${BLUE}PHASE 3: Cross-dataset validation (2-3 hours)${NC}"
echo "=========================================================================="
echo ""

python src/cross_validate.py

echo ""
echo -e "${GREEN}✓ Phase 3 complete: Cross-validation done${NC}"

# Phase 4: Grad-CAM
echo ""
echo "=========================================================================="
echo -e "${BLUE}PHASE 4: Generating Grad-CAM visualizations (3-4 hours)${NC}"
echo "=========================================================================="
echo ""

python src/gradcam.py --model all --samples 20

echo ""
echo -e "${GREEN}✓ Phase 4 complete: Grad-CAM visualizations generated${NC}"

# Final evaluation
echo ""
echo "=========================================================================="
echo -e "${BLUE}FINAL: Comprehensive evaluation${NC}"
echo "=========================================================================="
echo ""

python src/evaluate.py

# Summary
echo ""
echo "=========================================================================="
echo -e "${GREEN}  ✓ Training Pipeline Complete!${NC}"
echo "=========================================================================="
echo ""
echo "Results available in:"
echo "  - models/            Trained models (.h5 or .pt files)"
echo "  - results/metrics/   Performance metrics (JSON)"
echo "  - results/plots/     Visualizations (PNG)"
echo "  - results/gradcam/   Grad-CAM heatmaps"
echo ""
echo "Next steps:"
echo "  1. Review metrics: cat results/metrics/summary.json"
echo "  2. View training logs: tensorboard --logdir results/logs"
echo "  3. Launch web app: streamlit run app/streamlit_app.py"
echo ""
echo "=========================================================================="
