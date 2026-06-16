import os
import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from tensorflow import keras

# -----------------------------
# Relative path to model
# -----------------------------
MODEL_PATH = r"final-model.keras"

IMG_SIZE = 224
CLASS_NAMES = ["0", "1", "2", "3", "4"]

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
# Streamlit UI
# -----------------------------
st.title("Knee Osteoarthritis Classifier")
st.write("Upload an image to get prediction and Grad-CAM heatmap.")

uploaded_file = st.file_uploader(
    "Upload an image",
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
    predictions = model.predict([img_array])  # FIX: wrap in list

    # FIX: handle list output
    if isinstance(predictions, list):
        predictions = predictions[0]

    probs = predictions[0]
    predicted_class = np.argmax(probs)
    confidence = float(probs[predicted_class])

    st.subheader("Prediction")
    st.write(f"Predicted Class: **{CLASS_NAMES[predicted_class]}**")

    # -----------------------------
    # Confidence
    # -----------------------------
    st.write(f"Confidence: **{confidence*100:.2f}%**")
    st.progress(int(confidence * 100))

    # -----------------------------
    # Grad-CAM Heatmap
    # -----------------------------
    st.subheader("Original vs Grad-CAM Overlay")

    def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
        grad_model = tf.keras.models.Model(
            [model.inputs],
            [model.get_layer(last_conv_layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model([img_array])  # FIX

            # FIX: handle list output
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

    # Automatically detect last Conv2D layer
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

        # -----------------------------
        # Display
        # -----------------------------
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Original**")
            st.image(image, width="stretch")  # FIX

        with col2:
            st.write("**Grad-CAM**")
            st.image(Image.fromarray(superimposed_img), width="stretch")  # FIX


            # streamlit run src/interface/app.py