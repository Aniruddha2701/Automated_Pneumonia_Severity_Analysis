#!!!!  FOR GPU INstances, 
# set num_workers=4 and pin_memory=True in DataLoader for faster data loading.

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

def main():

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MODEL_PATH = "results/best_model_2.pth"
    TEST_DIR = "Dataset/Final/test"

    os.makedirs("results/metrics", exist_ok=True)

    # =========================
    # LOAD MODEL
    # =========================

    model = build_model(num_classes=3).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    all_preds = []
    all_labels = []

    # =========================
    # LOAD TEST DATA
    # =========================

    test_dataset = ImageFolder(
        root=TEST_DIR,
        transform=val_test_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,   # 🔥 IMPORTANT for Windows
        pin_memory=True
    )

    # =========================
    # PREDICT LOOP
    # =========================

    with torch.no_grad():

        for images, labels in test_loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

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
    # ACCURACY
    # =========================

    accuracy = (all_preds == all_labels).mean()
    print(f"\n Overall Accuracy: {accuracy*100:.2f}%")

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
    plt.tight_layout()
    plt.savefig("results/metrics/confusion_matrix.png")
    print("\n Confusion matrix saved.")
    plt.close()

if __name__ == "__main__":
    main()