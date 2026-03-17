# 🩻 Automated Pneumonia Severity Detection with Explainable AI

## 📌 Overview

This project presents a deep learning-based system for **automated pneumonia severity detection** from chest X-ray images.
In addition to classification, the system provides **visual explanations using Grad-CAM**, making the model's decisions interpretable.

The application is integrated with a **Streamlit-based user interface**, enabling real-time prediction and visualization.

---

## 🎯 Objectives

* Classify chest X-rays into:

  * Mild Pneumonia
  * Normal
  * Severe Pneumonia
* Enhance medical interpretability using Grad-CAM
* Provide a user-friendly interface for real-time predictions

---

## 🧠 Model Architecture

* CNN-based architecture using **DenseNet**
* Pretrained backbone for better feature extraction
* Fine-tuned for multi-class classification

---

## 🧪 Preprocessing Pipeline

* Image resizing to 224×224
* CLAHE (Contrast Limited Adaptive Histogram Equalization)
* Normalization using ImageNet statistics

---

## 🔥 Explainability with Grad-CAM

Grad-CAM (Gradient-weighted Class Activation Mapping) is used to:

* Highlight important regions in the X-ray
* Explain model predictions visually
* Improve trust in AI-based diagnosis

---

## 🖥️ User Interface

Built using **Streamlit**, the UI allows users to:

* Upload chest X-ray images
* View prediction results
* Visualize Grad-CAM heatmaps
* Download results

---

## 📊 Results

### 🔹 Prediction Output

* Model predicts severity class with confidence score

### 🔹 Visualization

* Side-by-side display:

  * Original Image
  * Grad-CAM Heatmap

---

## 📁 Project Structure

```
Automated_Pneumonia_Severity_Analysis/
│
├── app/
│   └── app.py
│
├── model/
│   └── model.py
│
├── preprocessing/
│   └── transforms.py
│
├── results/
│   ├── Heatmaps/
│   └── metrics/
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```
git clone <your-repo-link>
cd Automated_Pneumonia_Severity_Analysis
```

### 2️⃣ Create Virtual Environment

```
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```
streamlit run app/app.py
```

---

## 🧪 Usage

1. Upload a chest X-ray image
2. The model processes the image
3. Prediction is displayed (Mild / Normal / Severe)
4. Grad-CAM heatmap is generated
5. Download the result if needed

---

## ⚠️ Limitations

* Limited dataset size
* Not a substitute for professional medical diagnosis
* Performance depends on image quality

---

## 🚀 Future Work

* Use larger and diverse datasets
* Deploy as a web-based healthcare tool
* Integrate with hospital systems
* Improve localization of infected regions

---

## 👨‍💻 Author

**Aniruddha Maurya**

---

## 📜 License

This project is for academic and research purposes only.
