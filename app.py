import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import os
import time

# -------- CONFIGURATION --------
st.set_page_config(
    page_title="Brain Tumor Detection",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    /* Body background - Sky Blue */
    body {
        background-color: #87CEEB;
        color: #004080;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #004080;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Buttons - Bright Ocean Blue */
    .stButton>button {
        background-color: #1E90FF;
        color: white;
        border-radius: 8px;
        padding: 0.5em 1.2em;
        font-weight: bold;
        border: none;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1C86EE;
        color: white;
    }

    /* Input fields and text areas - Bright Ocean Blue */
    input, textarea, select {
        border: 1px solid #1E90FF;
        border-radius: 5px;
        padding: 0.4em;
        background-color: #1E90FF;
        color: white;
    }

    /* Sidebar background - Bright Ocean Blue */
    .css-1d391kg {
        background-color: #1E90FF;
        color: white;
    }

    /* Links */
    a {
        color: #004080;
    }

    /* Main content containers */
    .stCard, .stFrame {
        background-color: #1E90FF !important;
        color: white !important;
        border-radius: 10px;
        padding: 1em;
    }
    </style>
    """, unsafe_allow_html=True
)

# -------- LOAD MODEL --------
@st.cache_resource  # caches model so it loads only once
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "model", "best.pt")
    return YOLO(model_path)

model = load_model()

# -------- SIDEBAR --------
st.sidebar.title("Settings")
conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1, max_value=1.0, value=0.25, step=0.05
)

# -------- MAIN UI --------
st.title("Brain Tumor Detection (YOLO)")

uploaded_file = st.file_uploader(
    "Upload a brain MRI image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Convert uploaded file to OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.image(img_rgb, caption="Uploaded MRI", use_column_width=True)

    # Save temporary image for YOLO inference
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    save_folder = os.path.join(BASE_DIR, "predictions")
    os.makedirs(save_folder, exist_ok=True)
    temp_path = os.path.join(save_folder, f"input_{int(time.time())}.jpg")
    cv2.imwrite(temp_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

    # Run prediction
    results = model.predict(temp_path, conf=conf_threshold)

    # Check for tumor
    if len(results[0].boxes) == 0:
        st.success("No tumor detected")
        st.image(img_rgb, caption="Result Image", use_column_width=True)
    else:
        # Tumor detected
        annotated_img = results[0].plot()
        labels = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]
            labels.append(f"{label} ({conf:.2f})")
        label_text = ", ".join(labels)

        st.warning(f"Tumor detected: {label_text}")
        st.image(annotated_img, caption="Result Image", use_column_width=True)