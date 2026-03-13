import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os

from model.model import build_model
from preprocessing.transforms import train_transform, val_test_transform

# =========================
# DEVICE CONFIG
# =========================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", DEVICE)


# =========================
# TRAINING PARAMETERS
# =========================

BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3

CHECKPOINT_PATH = "results/checkpoint.pth"
BEST_MODEL_PATH = "results/best_model.pth"

os.makedirs("results", exist_ok=True)

# =========================
# DATSETS
# =========================

train_dataset = ImageFolder("Dataset/Final/train", transform=train_transform)
val_dataset = ImageFolder("Dataset/Final/val", transform=val_test_transform)

train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
val_loader = DataLoader(val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=4, 
    pin_memory=True
)

print("Class Mapping:", train_dataset.class_to_idx)

# =========================
# MODEL
# =========================

model = build_model(num_classes=3).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=LR)

# =========================
# CHECKPOINT RESUME
# =========================

start_epoch = 0
best_val_acc = 0

# 🔹 Resume from checkpoint if exists
if os.path.exists(CHECKPOINT_PATH):

    print("Loading checkpoint...")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location = DEVICE)

    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    start_epoch = checkpoint["epoch"]
    best_val_acc = checkpoint["best_val_acc"]

# =========================
# TRAINING LOOP
# =========================

for epoch in range(start_epoch, EPOCHS):

    model.train()

    train_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE) 
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total

# =========================
# VALIDATION
# =========================

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
    
        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Loss: {train_loss/len(train_loader):.4f} | "
        f"Train Acc: {train_acc:.2f}% | "
        f"Val Acc: {val_acc:.2f}%"
    )

# =========================
# SAVE CHECKPOINT
# =========================
    
    torch.save({
        "epoch": epoch + 1,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_acc": best_val_acc
    }, CHECKPOINT_PATH)

# =========================
# SAVE BEST MODEL
# =========================
    # 🔹 Save best model
    if val_acc > best_val_acc:
        
        best_val_acc = val_acc
        torch.save(model.state_dict(), "results/best_model.pth")

print("Training complete")