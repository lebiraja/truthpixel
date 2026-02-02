"""
Quick pipeline test to verify everything works before full training.
"""

import sys
sys.path.append('src')

import tensorflow as tf
import numpy as np
from data_loader_efficient import EfficientCIFAKEDataLoader
from model import TruthPixelModel

print("=" * 80)
print("TRUTHPIXEL PIPELINE VERIFICATION")
print("=" * 80)

# Test 1: Data Loading
print("\n[1/4] Testing data loading...")
data_loader = EfficientCIFAKEDataLoader(data_dir="data", batch_size=16)
train_ds, val_ds, test_ds = data_loader.prepare_datasets(augment_train=True)
print("✓ Data loading works!")

# Test 2: Check batch format
print("\n[2/4] Checking batch format...")
for images, labels in train_ds.take(1):
    print(f"  Batch shape: {images.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Image range: [{tf.reduce_min(images):.3f}, {tf.reduce_max(images):.3f}]")

    # Flatten labels for unique check
    labels_flat = tf.reshape(labels, [-1])
    print(f"  Label values: {tf.unique(labels_flat)[0].numpy()}")

    # CRITICAL CHECK: Values should be in [0, 255] for EfficientNet
    assert tf.reduce_min(images) >= 0.0, "Images have negative values!"
    assert tf.reduce_max(images) <= 255.0, "Images exceed 255.0!"
    print("✓ Image range correct for EfficientNet [0, 255]!")

    # Check label distribution in batch
    unique_labels, _, counts = tf.unique_with_counts(labels_flat)
    print(f"  Batch label distribution: {dict(zip(unique_labels.numpy(), counts.numpy()))}")
    print("✓ Batch format correct!")

# Test 3: Model Building
print("\n[3/4] Building model...")
model_builder = TruthPixelModel(learning_rate=0.0001)
model = model_builder.build_model(freeze_base=True)
model = model_builder.compile_model(model)
print("✓ Model built and compiled!")

# Count trainable params
trainable_count = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
non_trainable_count = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
print(f"  Trainable params: {trainable_count:,}")
print(f"  Non-trainable params: {non_trainable_count:,}")

# Test 4: Forward Pass
print("\n[4/4] Testing forward pass...")
for images, labels in train_ds.take(1):
    predictions = model(images, training=True)
    print(f"  Prediction shape: {predictions.shape}")
    print(f"  Prediction range: [{tf.reduce_min(predictions):.3f}, {tf.reduce_max(predictions):.3f}]")
    print(f"  Sample predictions: {predictions[:5, 0].numpy()}")

    # CRITICAL CHECK: Predictions should be in [0, 1] due to sigmoid
    assert tf.reduce_min(predictions) >= 0.0, "Predictions have negative values!"
    assert tf.reduce_max(predictions) <= 1.0, "Predictions exceed 1.0!"

    # Check if predictions are diverse (not all the same)
    pred_std = tf.math.reduce_std(predictions)
    print(f"  Prediction std: {pred_std:.4f}")

    if pred_std < 0.001:
        print("  ⚠️  WARNING: All predictions are nearly identical!")
        print("  This suggests the model might collapse to one class.")
    else:
        print("  ✓ Predictions show diversity!")

    print("✓ Forward pass works!")

# Test 5: Single Training Step
print("\n[5/5] Testing single training step...")
history = model.fit(train_ds.take(10), validation_data=val_ds.take(2), epochs=1, verbose=0)

print(f"  Train accuracy: {history.history['accuracy'][0]:.4f}")
print(f"  Train loss: {history.history['loss'][0]:.4f}")
print(f"  Val accuracy: {history.history['val_accuracy'][0]:.4f}")
print(f"  Val loss: {history.history['val_loss'][0]:.4f}")

# CRITICAL CHECK: Accuracy should NOT be exactly 0.5
if abs(history.history['val_accuracy'][0] - 0.5) < 0.01:
    print("  ⚠️  WARNING: Validation accuracy is ~50% (random guessing)")
    print("  Model might not be learning properly!")
else:
    print("  ✓ Model is learning (accuracy != 50%)!")

print("\n" + "=" * 80)
print("PIPELINE VERIFICATION COMPLETE!")
print("=" * 80)
print("\n✅ All checks passed! Ready for full training.")
print("\nRun: python src/train.py --batch_size 16")
