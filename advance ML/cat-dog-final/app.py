import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from keras.models import load_model
from keras.applications.mobilenet_v2 import preprocess_input
import os

st.set_page_config(page_title="Cat or Dog?", page_icon="🐾", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0f0f14; color: #e8e8f0; }
.hero-title { font-size: 2.8rem; font-weight: 700; letter-spacing: -0.03em;
              text-align: center; color: #ffffff; margin-bottom: 0.25rem; }
.hero-sub   { text-align: center; color: #888; font-size: 1rem; margin-bottom: 2.5rem; }
.result-box { border-radius: 14px; padding: 1.5rem 2rem; text-align: center;
              margin-top: 1.5rem; font-size: 1.8rem; font-weight: 700;
              letter-spacing: -0.02em; }
.result-cat { background: linear-gradient(135deg,#1e1b2e,#2d1f4e);
              border: 1.5px solid #7c5cbf; color: #c4a8ff; }
.result-dog { background: linear-gradient(135deg,#1b2120,#1a3328);
              border: 1.5px solid #3d9b6e; color: #6effc2; }
.conf-label { font-size:0.85rem; color:#888; margin-top:1rem;
              margin-bottom:0.3rem; text-align:left; }
.conf-bar-bg   { background:#2a2a36; border-radius:99px; height:10px;
                 width:100%; overflow:hidden; }
.conf-bar-fill { height:10px; border-radius:99px; }
.fill-cat { background: linear-gradient(90deg,#7c5cbf,#c4a8ff); }
.fill-dog { background: linear-gradient(90deg,#3d9b6e,#6effc2); }
.footer { text-align:center; color:#444; font-size:0.78rem; margin-top:3rem; }
</style>
""", unsafe_allow_html=True)

IMG_SIZE  = (128, 128)
THRESHOLD = 0.40   # Fix 3: lower than 0.5 to correct small-dog misclassification

@st.cache_resource(show_spinner=False)
def load_cnn_model():
    for path in [
        "saved_models/cat_dog_final.keras",
        "saved_models/best_model_phase2.keras",
        "saved_models/best_model_phase1.keras",
        "saved_models/cat_dog_savedmodel",
    ]:
        if os.path.exists(path):
            return load_model(path), path
    return None, None

def predict(model, pil_image: Image.Image):
    img = np.array(pil_image.convert("RGB"))
    img = cv2.resize(img, IMG_SIZE).astype(np.float32)
    img = preprocess_input(np.expand_dims(img, axis=0))   # [-1, 1] — matches training
    score = float(model.predict(img, verbose=0)[0][0])
    if score >= THRESHOLD:
        return "Dog", score
    else:
        return "Cat", 1 - score

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🐾 Cat or Dog?</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Upload a photo — the CNN decides.</div>', unsafe_allow_html=True)

with st.spinner("Loading model..."):
    model, model_path = load_cnn_model()

if model is None:
    st.error(
        "**Model not found.**\n\n"
        "Run `cat_dog_classifier.py` first to train & save the model. "
        "Expected at `saved_models/cat_dog_final.keras`"
    )
    st.stop()

st.caption(f"✅ Model loaded from `{model_path}`")

uploaded = st.file_uploader(
    label="Upload an image",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed"
)

if uploaded:
    pil_img = Image.open(uploaded)
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.image(pil_img, use_container_width=True, caption=uploaded.name)

    with st.spinner("Predicting..."):
        label, confidence = predict(model, pil_img)

    emoji     = "🐱" if label == "Cat" else "🐶"
    css_class = "result-cat" if label == "Cat" else "result-dog"
    fill_cls  = "fill-cat"   if label == "Cat" else "fill-dog"
    pct       = int(confidence * 100)

    st.markdown(f"""
    <div class="result-box {css_class}">
        {emoji} It's a <strong>{label}</strong>!
        <div style="font-size:1rem;font-weight:400;opacity:0.75;margin-top:0.25rem;">
            {pct}% confident
        </div>
    </div>
    <div class="conf-label">Confidence</div>
    <div class="conf-bar-bg">
        <div class="conf-bar-fill {fill_cls}" style="width:{pct}%"></div>
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="background:#1a1a24;border:1.5px dashed #333;border-radius:16px;
                padding:2.5rem 2rem;text-align:center;">
        <div style="font-size:2.5rem">📂</div>
        <div style="color:#aaa;margin-top:0.5rem;">
            Drag & drop or <strong style="color:#fff">browse</strong> a JPG / PNG / WEBP
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="footer">MobileNetV2 · fine-tuned on cats & dogs · built with Streamlit</div>',
            unsafe_allow_html=True)
