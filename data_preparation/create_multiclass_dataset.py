import os
import shutil
import random

# =========================
# CONFIGURATION
# =========================

SEED = 42
TARGET_PER_CLASS = 1500

POOL_DIR = "Dataset/Pool"
RAW_DIR = "Dataset/Raw"

NORMAL_SRC = os.path.join(POOL_DIR, "Normal")
PNEUMONIA_SRC = os.path.join(POOL_DIR, "Pneumonia")

NORMAL_DST = os.path.join(RAW_DIR, "Normal")
MILD_DST = os.path.join(RAW_DIR, "Mild")
SEVERE_DST = os.path.join(RAW_DIR, "Severe")

# =========================
# SET RANDOM SEED
# =========================

random.seed(SEED)

# =========================
# CREATE OUTPUT FOLDERS
# =========================

os.makedirs(NORMAL_DST, exist_ok=True)
os.makedirs(MILD_DST, exist_ok=True)
os.makedirs(SEVERE_DST, exist_ok=True)

# =========================
# LOAD FILE LISTS
# =========================

normal_files = os.listdir(NORMAL_SRC)
pneumonia_files = os.listdir(PNEUMONIA_SRC)

print(f"Total Normal available: {len(normal_files)}")
print(f"Total Pneumonia available: {len(pneumonia_files)}")

# =========================
# VALIDATION CHECK
# =========================

if len(normal_files) < TARGET_PER_CLASS:
    raise ValueError("Not enough Normal images to meet target.")

if len(pneumonia_files) < TARGET_PER_CLASS * 2:
    raise ValueError("Not enough Pneumonia images to split into Mild and Severe.")

# =========================
# SHUFFLE FILES
# =========================

random.shuffle(normal_files)
random.shuffle(pneumonia_files)

# =========================
# SELECT TARGET IMAGES
# =========================

selected_normal = normal_files[:TARGET_PER_CLASS]

selected_mild = pneumonia_files[:TARGET_PER_CLASS]
selected_severe = pneumonia_files[TARGET_PER_CLASS:TARGET_PER_CLASS * 2]

# =========================
# COPY FILES
# =========================

def copy_files(file_list, src_dir, dst_dir):
    for filename in file_list:
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, filename)
        shutil.copy2(src_path, dst_path)

print("Copying Normal images...")
copy_files(selected_normal, NORMAL_SRC, NORMAL_DST)

print("Copying Mild images...")
copy_files(selected_mild, PNEUMONIA_SRC, MILD_DST)

print("Copying Severe images...")
copy_files(selected_severe, PNEUMONIA_SRC, SEVERE_DST)

# =========================
# FINAL COUNT
# =========================

print("\nFinal Dataset Distribution:")
print("Normal:", len(os.listdir(NORMAL_DST)))
print("Mild:", len(os.listdir(MILD_DST)))
print("Severe:", len(os.listdir(SEVERE_DST)))

print("\n✅ Multi-class dataset created successfully.")