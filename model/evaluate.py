import torch
import numpy as np

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

import os

from model.model import build_model
from preprocessing.transforms import val_test_transform


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "results/best_model.pth"
TEST_DIR = "Dataset/Final/test"

os.makedirs("results/metrics", exist_ok=True)

# =========================
# LOAD MODEL
# =========================

model = build_model(num_classes=3).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location = DEVICE))
model.eval()

all_preds = []
all_labels = []

# =========================
# PREDICT LOOP
# =========================

with torch.no_grad():

    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images)

        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# =========================
# NUMPY ARRAYS CONVERSION
# =========================

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# =========================
# CLASSIFICATION REPORT
# =========================      
report = classification_report(
    all_labels, 
    all_preds, 
    target_names=test_loader.dataset.classes
    )

print("\n Classification Report: \n")
print(report)

with open("results/metrics/classification_report.txt", "w") as f:
    f.write(report)

# =========================
# CONFUSION MATRIX
# =========================

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, 
    annot=True, 
    fmt="d", 
    xticklabels=test_loader.dataset.classes, 
    yticklabels=test_loader.dataset.classes, 
    cmap="Blues" 
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")

plt.savefig("results/metrics/confusion_matrix.png")
print("\n Confusion matrix saved.")