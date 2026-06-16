import os
import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from tensorflow import keras

st.set_page_config(
    page_title="Knee OA Classifier",
    page_icon="🦴",
    layout="centered"
)

# -----------------------------
# Custom CSS for dark, modern look
# -----------------------------
st.markdown("""
<style>
.main-title {
    font-size: 2.4rem;
    font-weight: 700;
    color: #00D9C0;
    text-align: center;
    margin-bottom: 0.2rem;
    letter-spacing: -0.5px;
}
.subtitle {
    text-align: center;
    color: #9AA4B2;
    font-size: 1.05rem;
    margin-bottom: 2rem;
}
.card {
    background-color: #1B212C;
    border: 1px solid #2A323F;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35);
}
.result-grade {
    font-size: 3rem;
    font-weight: 800;
    color: #00D9C0;
    line-height: 1.1;
}
.result-label {
    color: #9AA4B2;
    font-size: 0.95rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.3rem;
}
.confidence-text {
    font-size: 1.1rem;
    font-weight: 600;
    color: #E6E9EF;
}
.section-header {
    color: #00D9C0;
    font-size: 1.2rem;
    font-weight: 700;
    margin-top: 0.5rem;
    margin-bottom: 0.8rem;
    border-left: 4px solid #00D9C0;
    padding-left: 0.6rem;
}
[data-testid="stFileUploader"] {
    background-color: #1B212C;
    border: 1px solid #2A323F;
    border-radius: 14px;
    padding: 1rem;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Relative path to model
# -----------------------------
MODEL_PATH = r"final-model.keras"

IMG_SIZE = 224
CLASS_NAMES = ["0", "1", "2", "3", "4"]
GRADE_LABELS = {
    "0": "Normal",
    "1": "Doubtful",
    "2": "Mild",
    "3": "Moderate",
    "4": "Severe"
}

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at: {MODEL_PATH}")
        st.stop()
    return keras.models.load_model(MODEL_PATH)

model = load_model()

# -----------------------------
# Header
# -----------------------------
st.markdown('<div class="main-title">🦴 Knee Osteoarthritis Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered KL grading with Grad-CAM interpretability</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload a knee X-ray image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    # -----------------------------
    # Preprocess
    # -----------------------------
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.asarray(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # -----------------------------
    # Prediction
    # -----------------------------
    predictions = model.predict([img_array])

    if isinstance(predictions, list):
        predictions = predictions[0]

    probs = predictions[0]
    predicted_class = np.argmax(probs)
    confidence = float(probs[predicted_class])
    grade = CLASS_NAMES[predicted_class]
    grade_name = GRADE_LABELS.get(grade, "")

    # -----------------------------
    # Result Card
    # -----------------------------
    st.markdown('<div class="section-header">Prediction Result</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="card">
        <div class="result-label">KL Grade</div>
        <div class="result-grade">{grade} &nbsp;<span style="font-size:1.4rem; color:#9AA4B2;">({grade_name})</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="card">
        <div class="result-label">Confidence</div>
        <div class="confidence-text">{confidence*100:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)
    st.progress(int(confidence * 100))

    # -----------------------------
    # Grad-CAM Heatmap
    # -----------------------------
    st.markdown('<div class="section-header">Grad-CAM Visualization</div>', unsafe_allow_html=True)

    def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
        grad_model = tf.keras.models.Model(
            [model.inputs],
            [model.get_layer(last_conv_layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model([img_array])

            if isinstance(predictions, list):
                predictions = predictions[0]

            class_index = tf.argmax(predictions[0])
            loss = predictions[:, class_index]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()

    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer.name
            break

    if last_conv_layer is not None:
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer)
        heatmap = cv2.resize(heatmap, (image.size[0], image.size[1]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        superimposed_img = cv2.addWeighted(
            np.array(image), 0.6, heatmap, 0.4, 0
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="result-label">Original</div>', unsafe_allow_html=True)
            st.image(image, width="stretch")

        with col2:
            st.markdown('<div class="result-label">Grad-CAM</div>', unsafe_allow_html=True)
            st.image(Image.fromarray(superimposed_img), width="stretch")
