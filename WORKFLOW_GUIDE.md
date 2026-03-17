# 🧭 Project Working Sequence Guide

## Automated Pneumonia Severity Analysis with Explainable AI

### (Deep Learning + Grad-CAM + Interactive UI)

---

## 1. Purpose of This Document

This document defines the **actual working pipeline** of the project as implemented in its final form.

It explains:

* execution order of operations
* interaction between components
* role of preprocessing, model, and explainability
* how the UI integrates the entire system

This serves as:

* academic reference
* viva explanation guide
* repository documentation

---

## 2. Updated High-Level Flow

Dataset Collection
→ Preprocessing Strategy
→ Model Training
→ Model Evaluation
→ Explainability Integration (Grad-CAM)
→ Deployment via UI (Streamlit)
→ Real-Time Inference Pipeline

---

## 3. Two Core Phases of the Project

### 🧪 Phase 1: Model Development (Offline)

This phase includes:

* dataset handling
* preprocessing
* training
* evaluation

👉 Happens once during development

---

### ⚡ Phase 2: Inference System (Online / UI)

This phase includes:

* image upload
* preprocessing
* prediction
* Grad-CAM visualization

👉 Happens every time user interacts with UI

---

## 4. Detailed Working Sequence

---

### 🔹 Step 1: Dataset Collection

A publicly available chest X-ray dataset is selected.

**Key considerations:**

* medical relevance
* class distribution (Mild, Normal, Severe)
* data quality

📌 No processing is done here.

---

### 🔹 Step 2: Data Organization

Dataset is structured into class-wise directories.

Example:

```id="l1t3y0"
Dataset/
  ├── Mild/
  ├── Normal/
  └── Severe/
```

---

### 🔹 Step 3: Preprocessing Strategy

Preprocessing is designed to improve image quality and consistency.

**Techniques used:**

* Resizing → (224 × 224)
* CLAHE → enhances contrast in X-rays
* Normalization → stabilizes model input

👉 This step defines how images are transformed before model input

---

### 🔹 Step 4: Model Training

A DenseNet-based CNN is trained on the processed dataset.

**Model learns:**

* texture patterns
* opacity regions
* structural abnormalities

📌 Feature extraction is automatic (deep learning)

---

### 🔹 Step 5: Model Evaluation

Model performance is measured using:

* Accuracy
* Classification Report
* Confusion Matrix

📁 Outputs:

```id="r7d3f2"
results/metrics/
```

---

### 🔹 Step 6: Explainability Integration (Grad-CAM)

Grad-CAM is integrated to visualize model decision-making.

**Function:**

* identifies important regions in image
* maps model attention
* improves interpretability

📌 This is a key enhancement over standard models

---

### 🔹 Step 7: Deployment via Streamlit UI

A user interface is built to integrate the full pipeline.

**User can:**

* upload X-ray image
* get prediction
* view heatmap

---

### 🔹 Step 8: Real-Time Inference Pipeline (CRITICAL)

This is the **actual runtime flow** when user uploads an image:

---

## 🔄 Real-Time Execution Flow

```id="1f6l7k"
User Upload Image
        ↓
Image Preprocessing (Resize + CLAHE + Normalize)
        ↓
Model Prediction (Mild / Normal / Severe)
        ↓
Confidence Score Calculation
        ↓
Grad-CAM Generation
        ↓
Overlay Heatmap on Image
        ↓
Display Results in UI
```

---

## 5. System Responsibility Split

| Component      | Responsibility                         |
| -------------- | -------------------------------------- |
| Student        | Data preparation, preprocessing design |
| Model          | Feature extraction & classification    |
| Grad-CAM       | Visual explanation of predictions      |
| UI (Streamlit) | User interaction & visualization       |

---

## 6. Key Clarifications

### 🔸 No Separate “Processed Dataset” Stored

Preprocessing is applied **dynamically using transforms**, not saved images.

👉 This improves:

* efficiency
* storage usage
* flexibility

---

### 🔸 Model Does Not Use Manual Features

All features are learned automatically by CNN.

---

### 🔸 Grad-CAM is Post-Prediction

It does NOT affect prediction
It only explains it

---

## 7. Final System Summary

The project is not just a classifier but an **Explainable AI System**.

```id="c5lgk8"
Input Image → Model → Prediction → Explanation → Visualization
```

---

## 8. Conclusion

This workflow represents the complete transformation:

```id="6rbn3g"
Raw X-ray → Processed Input → Learned Features → Prediction → Visual Explanation
```

The system ensures both:

* accuracy
* interpretability

making it suitable for academic and research use.

---

## End of Document
