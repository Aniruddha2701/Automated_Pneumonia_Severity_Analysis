# 📊 Dataset Preparation Module
## Automated Pneumonia Severity Analysis from Chest X-Ray Images
### using Deep Learning and Image Processing Techniques

---

<p align="center">
  🩺📸🧠  
</p>

> **Module Purpose:**  
> This module is responsible for organizing, labeling, and splitting the dataset
> to prepare it for supervised deep learning training.

---

## 🔹 1. Role of Dataset Preparation in the Pipeline

### 📌 Pipeline Context
Chest X-ray Images
→ Image Preprocessing
→ 📊 Dataset Preparation ← (YOU ARE HERE)
→ Deep Learning Model
→ Severity Classification
→ Evaluation

Dataset preparation acts as the **bridge** between preprocessing and model learning.

---

## 🔹 2. What This Module Does

✔ Defines **severity classes**  
✔ Organizes images into labeled folders  
✔ Splits data into **training and testing sets**  
✔ Ensures clean and structured input for the model  

⚠️ No learning happens here — this is **human-driven preparation**.

---

## 🔹 3. Severity Labeling Strategy (Academic & Safe)

### 🧪 Why Severity Approximation?
Most public chest X-ray datasets do **not provide explicit severity labels**.
Therefore, severity is **approximated for academic analysis**.

### ✅ Severity Classes Used
- **Normal**
- **Mild Pneumonia**
- **Severe Pneumonia**

📌 These labels are:
- visually guided
- used only for learning
- **not clinical diagnoses**

> 🔐 *Severity labeling is approximate and intended for academic study only.*

---

## 🔹 4. Dataset Organization Structure

Dataset/
│
├── Processed/
│ ├── Normal/
│ ├── Mild/
│ └── Severe/

✔ Clear class separation  
✔ Compatible with deep learning frameworks  
✔ Easy to explain in viva  

---

## 🔹 5. Train–Test Split Strategy

### 📈 Standard Practice
- **Training Set:** 70–80%
- **Testing Set:** 20–30%

This ensures:
- fair evaluation
- reduced overfitting
- proper generalization

> 📌 The testing data is never seen during training.

---

## 🔹 6. Files in This Module

### 📄 `split_data.py`
- Splits the dataset into training and testing sets
- Maintains class distribution

### 📄 `label_mapping.py`
- Defines class-to-label mapping
- Example:
  - Normal → 0
  - Mild → 1
  - Severe → 2

### 📄 `README.md`
- Explains dataset preparation logic
- Serves as documentation for viva and report

---

## 🔹 7. Responsibility Clarification (Viva Highlight)

## 👥 Responsibility Split: Human vs Model

This section clarifies the division of responsibilities between the developer
(student) and the deep learning model to ensure transparency and academic clarity.

| Task | Responsibility |
|---|---|
| Dataset selection | Student |
| Data organization & labeling | Student |
| Image preprocessing | Student |
| Data augmentation | Student |
| Dataset splitting (train/test) | Student |
| Feature extraction | Deep Learning Model |
| Pattern learning | Deep Learning Model |
| Severity classification | Deep Learning Model |
| Result interpretation | Student |

> 🔐 *The student is responsible for data preparation and system design,  
> while the deep learning model automatically learns features and patterns
> during training.*


---

## 🔹 8. Key Takeaway

> **Dataset preparation ensures that clean, well-labeled, and structured data
> is provided to the deep learning model for effective learning.**

---

## 🔹 9. Academic Note

- Public datasets are used
- Severity labels are approximate
- The system is **decision-support oriented**
- Not intended for real-world clinical deployment

---

✔ **Module Status:** Complete  
✔ **Pipeline Ready:** Yes  

---

🧠 *Next Module:*  
➡️ **Deep Learning Model (Model Definition & Training)**

---
