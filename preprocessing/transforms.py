import cv2
import numpy as np
from torchvision import transforms
from PIL import Image


# =========================
# CLAHE TRANSFORM
# =========================

class CLAHETransform:

    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid_size
        )

    def __call__(self, img):

        img = np.array(img)

        # Convert to grayscale if needed
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        img = self.clahe.apply(img)

        # Convert back to 3 channel
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        return Image.fromarray(img)


# =========================
# TRAIN TRANSFORM
# =========================

train_transform = transforms.Compose([

    transforms.Lambda(lambda img: img.convert("RGB")),

    CLAHETransform(),

    transforms.Resize((224, 224)),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================
# VALIDATION / TEST TRANSFORM
# =========================

val_test_transform = transforms.Compose([

    transforms.Lambda(lambda img: img.convert("RGB")),

    CLAHETransform(),

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


if __name__ == "__main__":
    print("Transforms module loaded successfully.")