import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys

from model.model import build_model
from preprocessing.transforms import val_test_transform

# =========================
# CONFIG
# =========================

MODEL_PATH = "results/best_model_2.pth"

if len(sys.argv) > 1:
    IMAGE_PATH = sys.argv[1]
else:
    IMAGE_PATH = "Dataset/Final/test/Mild/IM-0001-0001.jpeg"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["Mild", "Normal", "Severe"]

os.makedirs("results/Heatmaps", exist_ok=True)

# =========================
# LOAD MODEL
# =========================

model = build_model(num_classes=3).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# =========================
# HOOKS
# =========================

features = []

def forward_hook(module, input, output):
    features.clear()              # prevent accumulation
    features.append(output)

    # Only retain grad if output requires it (fixes issue with non-leaf tensors)
    if output.requires_grad:
        output.retain_grad()          # 🔥 KEY FIX

# Correct DenseNet layer
target_layer = model.features.norm5

# Register hook
hook_handle = target_layer.register_forward_hook(forward_hook)

# =========================
# LOAD IMAGE
# =========================

image = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = val_test_transform(image).unsqueeze(0).to(DEVICE)

# 🔥 IMPORTANT: enable gradients
input_tensor.requires_grad = True

# =========================
# FORWARD PASS
# =========================

output = model(input_tensor)
pred_class = output.argmax(dim=1).item()

# =========================
# BACKWARD PASS
# =========================

model.zero_grad()
output[0, pred_class].backward()

# =========================
# CHECK
# =========================

if len(features) == 0 or features[0].grad is None:
    raise RuntimeError("Gradients not captured — check target layer")

print("Features captured:", len(features))
print("Gradients captured: 1")

# =========================
# GENERATE HEATMAP
# =========================

grad = features[0].grad.cpu().numpy()[0]
feat = features[0].cpu().detach().numpy()[0]

weights = np.mean(grad, axis=(1, 2))

cam = np.zeros(feat.shape[1:], dtype=np.float32)

for i, w in enumerate(weights):
    cam += w * feat[i]

# ReLU
cam = np.maximum(cam, 0)

# Normalize safely
cam = cam / (cam.max() + 1e-8)

# Resize heatmap
heatmap = cv2.resize(cam, (224, 224))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# =========================
# OVERLAY
# =========================

original = cv2.imread(IMAGE_PATH)
original = cv2.resize(original, (224, 224))

overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

# =========================
# SAVE
# =========================

output_path = "results/Heatmaps/heatmap.jpg"
cv2.imwrite(output_path, overlay)

print(f"\nPrediction: {CLASS_NAMES[pred_class]}")
print(f"Heatmap saved at {output_path}")

# =========================
# DISPLAY
# =========================

plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.title(f"Grad-CAM: {CLASS_NAMES[pred_class]}")
plt.axis("off")
plt.show()

# =========================
# CLEANUP
# =========================

hook_handle.remove()