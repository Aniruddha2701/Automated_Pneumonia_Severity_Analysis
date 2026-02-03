# 🧪 Preprocessing Execution Guide
## Automated Pneumonia Severity Analysis from Chest X-Ray Images
### using Deep Learning and Image Processing Techniques

---

## 1. Purpose of This Document

This document explains the **actual execution flow** of image preprocessing,
describing how raw chest X-ray images are transformed into processed images
ready for deep learning training.

This guide focuses on **what happens first, next, and last** during preprocessing.

---

## 2. Input and Output Locations

### 📥 Input Directory
    
    Dataset/Raw/

Contains original chest X-ray images.  
⚠️ These images are never modified.

### 📤 Output Directory

    Dataset/Processed/

Contains processed, model-ready images.

---

## 3. Preprocessing Execution Flow
Load Raw Image
→ Resize Image
→ Normalize Pixel Values
→ Reduce Noise
→ Enhance Contrast
→ Save Processed Image

This flow is applied **consistently to all images**.

---

## 4. Execution Steps Explained

### Step 1: Image Loading
- Raw images are read from disk
- Converted to grayscale if required

Purpose:
- Simplifies processing
- Matches medical imaging characteristics

---

### Step 2: Image Resizing
- All images are resized to a fixed resolution

Purpose:
- Ensures uniform input dimensions
- Required for deep learning models

---

### Step 3: Pixel Normalization
- Pixel values are scaled to a standard range

Purpose:
- Improves numerical stability
- Supports efficient model training

---

### Step 4: Noise Reduction
- Minor artifacts and noise are reduced

Purpose:
- Enhances clarity of lung structures
- Prevents learning from noise patterns

---

### Step 5: Contrast Enhancement
- Image contrast is improved

Purpose:
- Highlights infection-related patterns
- Improves visibility of lung opacities

---

### Step 6: Saving Processed Images
- Processed images are saved to disk
- Folder structure is preserved

Purpose:
- Maintains separation between raw and processed data
- Ensures reproducibility

---

## 5. Responsibility Split: Human vs Model

| Task | Responsibility |
|---|---|
| Loading raw images | Student |
| Image resizing | Student |
| Pixel normalization | Student |
| Noise reduction | Student |
| Contrast enhancement | Student |
| Saving processed images | Student |
| Feature learning | Deep Learning Model |
| Pattern recognition | Deep Learning Model |

> 🔐 *All preprocessing steps are explicitly executed by the student.
> The deep learning model does not learn or modify preprocessing operations.*

---

## 6. Key Clarification

> **Preprocessing is a deterministic transformation, not a learning process.**

Only after preprocessing is complete does the deep learning model begin learning.

---

## 7. Status

✔ Preprocessing execution defined  
✔ Raw and processed data clearly separated  
✔ Ready for dataset preparation and model training  

---

## End of Document
