# AuthentiScan Evaluation Guide

This guide explains how to evaluate AuthentiScan models, interpret metrics, and use Grad-CAM for explainability.

## 📊 Evaluation Metrics

AuthentiScan calculates comprehensive metrics to assess model performance across different datasets and generators.

### key Metrics

1. **Accuracy**: Overall percentage of correctly classified images.
   - Target: >95% on training domain, >88% cross-domain.

2. **Precision**: Fraction of detected fakes that are actually fake.
   - High precision = Few false alarms (Real images flagged as Fake).

3. **Recall**: Fraction of actual fakes that were detected.
   - High recall = Few missed fakes.

4. **F1-Score**: Harmonic mean of Precision and Recall.
   - Best single metric for balanced performance.

5. **AUC-ROC**: Area Under the Receiver Operating Characteristic curve.
   - Measures discrimination ability independent of threshold.
   - 0.5 = Random guessing, 1.0 = Perfect separation.

6. **Confusion Matrix**:
   - **True Positives (TP)**: Fake classified as Fake
   - **True Negatives (TN)**: Real classified as Real
   - **False Positives (FP)**: Real classified as Fake (False Alarm)
   - **False Negatives (FN)**: Fake classified as Real (Missed Detection)

---

## 🔄 Cross-Validation Interpretation

The `cross_validate.py` script generates a performance matrix showing how well models generalize.

### Example Matrix

| Model Trained On ↓ / Tested On → | GenImage | CIFAKE | Faces | Combined |
|----------------------------------|----------|--------|-------|----------|
| **GenImage Model**               | **96.5%**| 82.1%  | 78.4% | 85.6%    |
| **CIFAKE Model**                 | 75.3%    | **94.8%**| 68.2% | 79.4%    |
| **Faces Model**                  | 72.1%    | 65.4%  | **98.2%**| 78.5%    |
| **Combined Model**               | 94.2%    | 92.1%  | 96.5% | **94.8%**|

### Interpretation
- **Diagonal (Bold)**: In-domain performance (usually highest).
- **Off-Diagonal**: Cross-domain generalization (lower is expected).
- **Combined Model Row**: Should show high performance across ALL columns, demonstrating robust generalization.
- **Generalization Gap**: Difference between diagonal and off-diagonal values (lower gap is better).

---

## 🧠 Grad-CAM Explainability

Grad-CAM (Gradient-weighted Class Activation Mapping) visualizes which parts of an image contributed most to the model's decision.

### How to Run
```bash
python src/gradcam.py --model combined --samples 20
```

### Interpreting Heatmaps

- **Red/Hot Regions**: High importance (features the model used).
- **Blue/Cold Regions**: Low importance.

#### What to Look For:
1. **Deepfakes (Faces)**:
   - **Good**: Focus on eyes, mouth, hair boundaries, or background anomalies.
   - **Bad**: Focus on irrelevant background (shortcut learning).

2. **GAN/Diffusion Images**:
   - **Good**: Focus on artifacts, unnatural textures, or structural inconsistencies.
   - **Bad**: Focus on common objects present in only one class.

### Output Location
Visualizations are saved to `results/gradcam/{model_name}/{dataset}/`.

---

## 📈 Generating Reports

Run the full evaluation pipeline:
```bash
python src/evaluate.py
```

This generates:
1. **`results/metrics/summary.json`**: All numerical metrics.
2. **`results/plots/roc_curves.png`**: ROC curves for all models.
3. **`results/plots/confusion_matrices.png`**: Side-by-side confusion matrices.
4. **`results/cross_validation/heatmap.png`**: Visual representation of the cross-validation matrix.

---

## 🖥️ Streamlit Dashboard

For interactive evaluation:
```bash
streamlit run app/streamlit_app.py
```

- **Upload Image**: Test individual images.
- **Model Comparison**: Run all models on the same image side-by-side.
- **Batch Processing**: Evaluate a folder of images and download CSV results.
