import torch
from PIL import Image
import sys

from model.model import build_model
from preprocessing.transforms import val_test_transform


# =========================
# CONFIG
# =========================

MODEL_PATH = "results/best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["Mild", "Normal", "Severe"]


# =========================
# CHECK INPUT
# =========================

if len(sys.argv) < 2:
    print("❌ Please provide image path")
    print("Usage: python -m model.predict image.jpg")
    sys.exit()

image_path = sys.argv[1]


# =========================
# LOAD MODEL
# =========================

model = build_model(num_classes=3).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()


# =========================
# LOAD IMAGE
# =========================

image = Image.open(image_path).convert("RGB")

image = val_test_transform(image)

image = image.unsqueeze(0).to(DEVICE)


# =========================
# PREDICTION
# =========================

with torch.no_grad():

    outputs = model(image)

    probabilities = torch.softmax(outputs, dim=1)

    confidence, predicted = torch.max(probabilities, dim=1)


predicted_class = CLASS_NAMES[predicted.item()]
confidence_score = confidence.item() * 100


# =========================
# OUTPUT
# =========================

print("\n✅ Prediction Result")
print("------------------------")

print("\n📊 Class Probabilities:")
for i, prob in enumerate(probabilities[0]):
    print(f"{CLASS_NAMES[i]}: {prob.item()*100:.2f}%")