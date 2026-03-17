import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import sys
import os

# =========================
# FIX IMPORT PATH
# =========================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.model import build_model
from preprocessing.transforms import val_test_transform

# =========================
# CONFIG
# =========================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["Mild", "Normal", "Severe"]
MODEL_PATH = "results/best_model_2.pth"

# =========================
# LOAD MODEL
# =========================

@st.cache_resource
def load_model():
    model = build_model(num_classes=3).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

model = load_model()

# =========================
# GRAD-CAM FUNCTION
# =========================

def generate_gradcam(model, image):

    features = []

    def forward_hook(module, input, output):
        features.clear()
        features.append(output)
        if output.requires_grad:
            output.retain_grad()

    target_layer = model.features.norm5
    hook = target_layer.register_forward_hook(forward_hook)

    input_tensor = val_test_transform(image).unsqueeze(0).to(DEVICE)
    input_tensor.requires_grad = True

    output = model(input_tensor)

    probs = torch.softmax(output, dim=1)
    confidence, pred_class = torch.max(probs, 1)

    pred_class = pred_class.item()
    confidence = confidence.item()

    model.zero_grad()
    output[0, pred_class].backward()

    if len(features) == 0 or features[0].grad is None:
        hook.remove()
        raise RuntimeError("Grad-CAM failed")

    grad = features[0].grad.cpu().numpy()[0]
    feat = features[0].detach().cpu().numpy()[0]

    weights = np.mean(grad, axis=(1, 2))

    cam = np.zeros(feat.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * feat[i]

    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)

    heatmap = cv2.resize(cam, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original = cv2.cvtColor(np.array(image.resize((224, 224))), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    hook.remove()

    return overlay, pred_class, confidence

# =========================
# UI CONFIG
# =========================

st.set_page_config(page_title="Pneumonia Detection", layout="wide")

st.markdown(
    "<h1 style='text-align: center;'>🩻 Pneumonia Severity Detection</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center;'>Upload a chest X-ray to get prediction + explanation</p>",
    unsafe_allow_html=True
)

# =========================
# UPLOAD
# =========================

uploaded_file = st.file_uploader("Upload Chest X-ray", type=["jpg", "jpeg", "png"])

# =========================
# PROCESS
# =========================

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    with st.spinner("🔍 Analyzing image..."):

        heatmap, pred_class, confidence = generate_gradcam(model, image)

    # =========================
    # DISPLAY RESULTS
    # =========================

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original Image", use_column_width=True)

    with col2:
        st.image(
            cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB),
            caption="Grad-CAM Heatmap",
            use_column_width=True
        )

    # =========================
    # METRICS
    # =========================

    st.markdown("---")

    colA, colB = st.columns(2)

    with colA:
        st.metric("Prediction", CLASS_NAMES[pred_class])

    with colB:
        st.metric("Confidence", f"{confidence*100:.2f}%")

    # =========================
    # DOWNLOAD BUTTON
    # =========================

    st.markdown("---")

    _, buffer = cv2.imencode(".jpg", heatmap)

    st.download_button(
        label="📥 Download Heatmap",
        data=buffer.tobytes(),
        file_name="gradcam_result.jpg",
        mime="image/jpeg"
    )