import cv2
import numpy as np
from torchvision import transforms
from PIL import Image


# =========================
# CLAHE FUNCTION
# =========================

def apply_clahe(img):
    img_np = np.array(img)

    # Convert to grayscale if needed
    if len(img_np.shape) == 3:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_np)

    # Convert back to 3-channel
    img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)

    return Image.fromarray(img_clahe)


# =========================
# TRAIN TRANSFORM
# =========================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda img: apply_clahe(img)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# =========================
# VALIDATION / TEST TRANSFORM
# =========================

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda img: apply_clahe(img)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])