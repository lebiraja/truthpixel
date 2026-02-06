# AuthentiScan Training Guide

This guide explains the 4-phase training methodology used in AuthentiScan to achieve robust AI-generated image detection.

## 🏗️ 4-Phase Training Strategy

AuthentiScan employs a novel 4-phase training approach designed to maximize performance and generalization while ensuring explainability.

### Phase 1: Baseline Models (12-18 hours)
**Goal:** Establish baseline performance for each dataset individually.

- **Process:**
  1. Train separate EfficientNet-B0 models for GenImage, CIFAKE, and Faces.
  2. Use standard augmentation (HorizontalFlip, Rotate, ColorJitter).
  3. Train for 20 epochs with EarlyStopping (patience=7).
  4. Save best model based on validation accuracy.

- **Outcome:** 3 specialized models (`genimage_baseline.pt`, `cifake_baseline.pt`, `faces_baseline.pt`).

### Phase 2: Combined Model (18-24 hours)
**Goal:** Create a unified model that generalizes across all domains.

- **Process:**
  1. Initialize with ImageNet pretrained weights.
  2. Train on **all 3 datasets** simultaneously using `WeightedRandomSampler`.
  3. **Weights:** GenImage (44%), CIFAKE (18%), Faces (38%).
  4. **Freezing Strategy:**
     - Epochs 1-5: Freeze backbone, train only head.
     - Epochs 6-25: Unfreeze last 20 layers of backbone.
  5. Checkpoint every 5 epochs.

- **Outcome:** A single robust model (`combined_model.pt`) capable of detecting fakes from multiple generators.

### Phase 3: Cross-Dataset Validation (2-3 hours)
**Goal:** Evaluate true generalization capability.

- **Process:**
  1. Test **GenImage** model on CIFAKE & Faces.
  2. Test **CIFAKE** model on GenImage & Faces.
  3. Test **Faces** model on GenImage & CIFAKE.
  4. Test **Combined** model on all datasets.

- **Outcome:** A cross-validation matrix quantifying generalization gaps.

### Phase 4: Grad-CAM Explainability (3-4 hours)
**Goal:** Visualize model decision-making.

- **Process:**
  1. Generate Class Activation Maps (CAM) for the last convolutional layer.
  2. Process 20 random samples (10 Real, 10 Fake) for each model/dataset pair.
  3. Overlay heatmaps on original images.

- **Outcome:** Visual evidence of what features the models are focusing on (e.g., eyes in deepfakes, artifacts in SD images).

---

## ⚙️ Configuration

Training parameters are defined in `config.yaml`.

### Key Settings

```yaml
# Hardware
hardware:
  gpu: "RTX 4050 Laptop"
  vram_gb: 6
  mixed_precision: true  # FP16 training

# Data
data:
  batch_size: 16  # Adjust based on VRAM
  num_workers: 4

# Training
training:
  learning_rate: 0.0001
  optimizer: "adam"
  weight_decay: 0.0001
```

### Hyperparameter Tuning

- **Batch Size:**
  - 6GB VRAM → 16
  - 8GB VRAM → 32
  - 12GB+ VRAM → 64

- **Learning Rate:**
  - Default: 1e-4
  - If loss oscillates: Reduce to 1e-5
  - If loss stagnates: Increase to 5e-4

---

## 🚀 Running Training

### Full Pipeline (Recommended)
Runs all 4 phases sequentially.
```bash
bash scripts/train_all.sh
```

### Individual Phases

**Phase 1: Baseline**
```bash
python src/train_baseline.py --dataset genimage
python src/train_baseline.py --dataset cifake
python src/train_baseline.py --dataset faces
```

**Phase 2: Combined**
```bash
python src/train_combined.py
```

**Phase 3: Cross-Validation**
```bash
python src/cross_validate.py
```

**Phase 4: Grad-CAM**
```bash
python src/gradcam.py --model all --samples 20
```

---

## 📊 Monitoring

Use TensorBoard to monitor training progress in real-time.

```bash
tensorboard --logdir results/logs
```

**Metrics to Watch:**
1. **Loss (Train vs Val):** Ensure validation loss decreases along with training loss. Divergence indicates overfitting.
2. **Accuracy:** Should reach >90% within 5-10 epochs.
3. **Learning Rate:** Check if `ReduceLROnPlateau` is triggering correctly.

---

## 💾 Checkpoints & Resume

- **Checkpoints:** Saved every 5 epochs in `models/checkpoints/`.
- **Best Model:** Saved as `*_best.pt` in `models/{baseline,combined}/`.

**To resume training (not yet implemented in CLI, requires code modification):**
Load the checkpoint state dict in the training script.

```python
checkpoint = torch.load('models/checkpoints/checkpoint_epoch_10.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
```
