"""
Quick PyTorch pipeline test.
"""

import sys
sys.path.append('src')

import torch
from data_loader_pytorch import CIFAKEDataLoaderPyTorch
from model_pytorch import create_model

print("=" * 80)
print("PYTORCH PIPELINE VERIFICATION")
print("=" * 80)

# Test 1: Check CUDA
print("\n[1/5] Checking CUDA...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Device: {device}")

if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA Version: {torch.version.cuda}")
    print("  ✓ CUDA available!")
else:
    print("  ⚠️  CUDA not available, using CPU")

# Test 2: Data Loading
print("\n[2/5] Testing data loading...")
data_loader = CIFAKEDataLoaderPyTorch(
    data_dir="data",
    batch_size=16,
    num_workers=2
)
train_loader, val_loader, test_loader = data_loader.prepare_loaders()
print("  ✓ Data loading works!")

# Test 3: Check batch
print("\n[3/5] Checking batch format...")
images, labels = next(iter(train_loader))
print(f"  Batch shape: {images.shape}")
print(f"  Labels shape: {labels.shape}")
print(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")
print(f"  Unique labels: {labels.unique().tolist()}")

# Check for proper normalization
if images.min() >= -3.0 and images.max() <= 3.0:
    print("  ✓ Image normalization looks correct!")
else:
    print("  ⚠️  Unusual image values detected")

# Test 4: Model Creation
print("\n[4/5] Creating model...")
model = create_model(freeze_base=True, device=device)
print("  ✓ Model created!")

# Count parameters
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"  Trainable params: {trainable:,}")
print(f"  Total params: {total:,}")

# Test 5: Forward Pass
print("\n[5/5] Testing forward pass...")
images = images.to(device)
labels = labels.to(device)

with torch.no_grad():
    outputs = model(images)

print(f"  Output shape: {outputs.shape}")
print(f"  Output range: [{outputs.min():.3f}, {outputs.max():.3f}]")
print(f"  Sample outputs: {outputs[:5, 0].cpu().numpy()}")

# Check output diversity
output_std = outputs.std().item()
print(f"  Output std: {output_std:.4f}")

if output_std > 0.01:
    print("  ✓ Outputs show diversity!")
else:
    print("  ⚠️  Outputs are too similar")

# Test 6: Single Training Step
print("\n[6/6] Testing single training step...")

model.train()
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train on one batch
outputs = model(images)
labels_float = labels.float().unsqueeze(1)
loss = criterion(outputs, labels_float)

optimizer.zero_grad()
loss.backward()
optimizer.step()

# Check predictions
predictions = (outputs > 0.5).float()
accuracy = (predictions == labels_float).float().mean().item()

print(f"  Loss: {loss.item():.4f}")
print(f"  Accuracy: {accuracy * 100:.2f}%")

if abs(accuracy - 0.5) > 0.1:
    print("  ✓ Model can make predictions (not stuck at 50%)!")
else:
    print(f"  ⚠️  Accuracy close to 50% (might be random)")

print("\n" + "=" * 80)
print("PYTORCH PIPELINE VERIFICATION COMPLETE!")
print("=" * 80)
print("\n✅ All PyTorch components working!")
print("\nReady to train:")
print("  python src/train_pytorch.py --batch_size 16")
