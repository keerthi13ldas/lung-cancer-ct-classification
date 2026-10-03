import os

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from huggingface_hub import hf_hub_download
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input

MODEL_FILE = "best_lung_cancer_resnet50_finetuned.keras"
LOCAL_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                MODEL_FILE)
HF_REPO_ID = "Keerthanldas/lung-cancer-resnet50"
CLASS_NAMES = ["Benign", "Malignant", "Normal"]
CLASS_COLORS = {"Benign": "#e0a030", "Malignant": "#d64545", "Normal": "#2e9e6b"}
IMG_SIZE = (224, 224)

st.set_page_config(page_title="Lung Cancer CT Classifier",
                   page_icon="🫁", layout="centered")

st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {padding-top: 1.5rem; max-width: 900px;}

    .header {
        background: linear-gradient(135deg, #0b2545 0%, #13508a 100%);
        padding: 28px 32px; border-radius: 14px; margin-bottom: 24px;
        text-align: center;
    }
    .header h1 {color: #ffffff; margin: 0; font-size: 2.2rem; font-weight: 700;}
    .header p {color: #cfe0f5; margin: 8px 0 0 0; font-size: 1.05rem;}

    .section-title {
        font-size: 0.85rem; text-transform: uppercase;
        letter-spacing: 0.08em; opacity: 0.7; margin: 22px 0 8px 0;
    }
    .result {
        border-radius: 12px; padding: 22px 24px; text-align: center;
        border: 1px solid rgba(128,128,128,0.25);
    }
    .result .tag {font-size: 0.8rem; text-transform: uppercase;
                  letter-spacing: 0.08em; opacity: 0.75;}
    .result .label {font-size: 2.4rem; font-weight: 700; margin: 4px 0;}
    .result .conf {font-size: 1.1rem; font-weight: 600;}

    .disclaimer {
        background: rgba(214,69,69,0.10); border-left: 5px solid #d64545;
        border-radius: 8px; padding: 12px 16px; font-size: 0.9rem;
        margin-top: 26px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    # Use the local file if it exists (running on your own PC)
    if os.path.exists(LOCAL_MODEL_PATH):
        return tf.keras.models.load_model(LOCAL_MODEL_PATH)
    # Otherwise download it from the private Hugging Face model repo
    path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=MODEL_FILE,
        token=st.secrets["HF_TOKEN"],
    )
    return tf.keras.models.load_model(path)


# 1. Header
st.markdown(
    "<div class='header'>"
    "<h1>🫁 Lung Cancer CT Classification</h1>"
    "<p>AI-Based CT Image Classification</p>"
    "</div>",
    unsafe_allow_html=True,
)

model = load_model()

# 2. Image upload section
st.markdown("<div class='section-title'>1. Upload CT image</div>",
            unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Drag and drop a CT scan here, or click to browse",
    type=["jpg", "jpeg", "png"],
)

if uploaded is None:
    st.session_state.pop("result", None)
    st.session_state.pop("file_key", None)
    st.info("Supported formats: JPG, JPEG and PNG.")
else:
    file_key = f"{uploaded.name}-{uploaded.size}"
    if st.session_state.get("file_key") != file_key:
        st.session_state["file_key"] = file_key
        st.session_state.pop("result", None)

    image = Image.open(uploaded).convert("RGB")

    # 3. Image preview
    st.markdown("<div class='section-title'>2. Image preview</div>",
                unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.image(image, caption=uploaded.name, use_container_width=True)

    # 4. Prediction button
    st.markdown("<div class='section-title'>3. Run prediction</div>",
                unsafe_allow_html=True)
    if st.button("🔍 Predict", type="primary", use_container_width=True):
        with st.spinner("Analysing image..."):
            arr = np.array(image.resize(IMG_SIZE), dtype="float32")
            arr = preprocess_input(np.expand_dims(arr, axis=0))
            probs = model.predict(arr, verbose=0)[0]
        st.session_state["result"] = [float(p) for p in probs]

    # 5. Prediction result and 6. Probability chart
    if "result" in st.session_state:
        probs = st.session_state["result"]
        idx = int(np.argmax(probs))
        label = CLASS_NAMES[idx]
        confidence = probs[idx] * 100
        color = CLASS_COLORS[label]

        st.markdown("<div class='section-title'>4. Prediction result</div>",
                    unsafe_allow_html=True)
        st.markdown(
            f"<div class='result' style='background:{color}1f;"
            f"border-left:8px solid {color};'>"
            "<div class='tag'>Predicted class</div>"
            f"<div class='label' style='color:{color};'>{label}</div>"
            f"<div class='conf'>Confidence: {confidence:.2f}%</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        if confidence < 60:
            st.warning("Low confidence. Interpret this result with caution.")

        st.markdown("<div class='section-title'>5. Class probabilities</div>",
                    unsafe_allow_html=True)
        df = pd.DataFrame(
            {"Probability (%)": [p * 100 for p in probs]},
            index=CLASS_NAMES,
        )
        st.bar_chart(df, y="Probability (%)", color="#13508a")
        cols = st.columns(3)
        for col, name, p in zip(cols, CLASS_NAMES, probs):
            col.metric(name, f"{p * 100:.2f}%")

# 7. Disclaimer
st.markdown(
    "<div class='disclaimer'><b>Disclaimer:</b> This system is for academic "
    "and research purposes only. It does not provide a medical diagnosis and "
    "must not be used for clinical decisions. Consult a qualified medical "
    "professional.</div>",
    unsafe_allow_html=True,
)