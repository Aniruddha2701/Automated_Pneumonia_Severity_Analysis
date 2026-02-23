import os
import shutil
import random

# =========================
# CONFIGURATION
# =========================

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RAW_DIR = "Dataset/Raw"
FINAL_DIR = "Dataset/Final"

CLASSES = ["Normal", "Mild", "Severe"]

random.seed(SEED)

# =========================
# CREATE FINAL FOLDERS
# =========================

for split in ["train", "val", "test"]:
    for cls in CLASSES:
        os.makedirs(os.path.join(FINAL_DIR, split, cls), exist_ok=True)

# =========================
# SPLIT FUNCTION
# =========================

def split_class(class_name):
    class_path = os.path.join(RAW_DIR, class_name)
    images = os.listdir(class_path)

    random.shuffle(images)

    total = len(images)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_imgs = images[:train_end]
    val_imgs = images[train_end:val_end]
    test_imgs = images[val_end:]

    print(f"\nClass: {class_name}")
    print(f"Total: {total}")
    print(f"Train: {len(train_imgs)}")
    print(f"Val: {len(val_imgs)}")
    print(f"Test: {len(test_imgs)}")

    for img in train_imgs:
        shutil.copy2(
            os.path.join(class_path, img),
            os.path.join(FINAL_DIR, "train", class_name, img)
        )

    for img in val_imgs:
        shutil.copy2(
            os.path.join(class_path, img),
            os.path.join(FINAL_DIR, "val", class_name, img)
        )

    for img in test_imgs:
        shutil.copy2(
            os.path.join(class_path, img),
            os.path.join(FINAL_DIR, "test", class_name, img)
        )

# =========================
# EXECUTE SPLIT
# =========================

for cls in CLASSES:
    split_class(cls)

print("\n✅ 70/15/15 stratified split completed successfully.")