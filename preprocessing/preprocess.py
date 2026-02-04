import os
import cv2
import numpy as np
from tqdm import tqdm # what is tqdm: A library for displaying progress bars in loops.

# Paths
RAW_DIR = "Dataset/Raw"
PROCESSED_DIR = "Dataset/Processed"

# Preprocessing parameters
IMG_SIZE = 224

def preprocess_image(img_path):
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

    # Noise reduction
    denoised = cv2.GaussianBlur(resized, (3, 3), 0)

    # Contrast enhancement
    enhanced = cv2.equalizeHist(denoised)

    # Normalize to [0, 1]
    normalized = enhanced / 255.0

    return normalized


def process_category(category):
    input_dir = os.path.join(RAW_DIR, category)
    output_dir = os.path.join(PROCESSED_DIR, category)

    os.makedirs(output_dir, exist_ok=True)

    for img_name in tqdm(os.listdir(input_dir), desc=f"Processing {category}"):
        img_path = os.path.join(input_dir, img_name)
        processed_img = preprocess_image(img_path)

        if processed_img is not None:
            save_path = os.path.join(output_dir, img_name)
            cv2.imwrite(save_path, (processed_img * 255).astype(np.uint8))


def main():
    for category in ["Normal", "Pneumonia"]:
        process_category(category)

    print("✅ Preprocessing complete. Processed images saved.")


if __name__ == "__main__":
    main()
