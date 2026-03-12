import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from model import build_model
from torch.utils.data import DataLoader
from preprocessing.transforms import train_transform, val_test_transform

# 2 . DEVICE CONFIGURATION
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3 . DATASET AND DATALOADER
train_dataset = ImageFolder("Dataset/Final/train", transform=train_transform)
val_dataset = ImageFolder("Dataset/Final/val", transform=val_test_transform)

#test_dataset = ImageFolder("Dataset/Final/test", transform=val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

#test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 4 . MODEL, LOSS, OPTIMIZER
model = build_model(num_classes=3).to(DEVICE)
criterion = nn.CrossEntropyLoss()  
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 5 . TRAINING LOOP
for epoch in range(10):
    model.train()
    running_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")