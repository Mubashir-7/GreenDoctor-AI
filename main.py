# ---------------------------------------------------------
# Plant Disease Classifier — Modern UI with Grad-CAM & Top-K
# ---------------------------------------------------------
import os
import io
import json
import base64
from typing import Tuple, Dict, List

import numpy as np
from PIL import Image, ImageOps

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

# ----------------------------
# Page config & custom styling
# ----------------------------
st.set_page_config(
    page_title="🌿 Plant Disease Classifier",
    page_icon="🌱",
    layout="wide",
)

# Glassmorphism + subtle animations
st.markdown(
<style>
/* Background gradient */
.stApp {
  background: radial-gradient(1200px 800px at 10% 10%, #ecfff6 0%, #f7fbff 40%, #ffffff 100%);
}

/* Glass cards */
.card {
  background: rgba(255,255,255,0.55);
  border-radius: 20px;
  padding: 1.25rem 1.5rem;
  box-shadow: 0 8px 30px rgba(0,0,0,0.07);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255,255,255,0.6);
  transition: transform .2s ease, box-shadow .2s ease;
}
.card:hover { transform: translateY(-3px); box-shadow: 0 14px 40px rgba(0,0,0,0.10); }

/* Buttons */
.stButton>button {
  border-radius: 12px;
  padding: .6rem 1rem;
  font-weight: 600;
}

/* Progress bars thin & rounded */
[data-testid="stProgressBar"] > div > div {
  height: 10px; border-radius: 9999px;
}

/* Hide default footer */
footer {visibility: hidden;}
</style>
, unsafe_allow_html=True)

# ---------------
# Paths & loading
# ---------------
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(WORKING_DIR, "trained_model", "plant_disease_prediction_model.h5")
CLASS_IDX_PATH = os.path.join(WORKING_DIR, "class_indices.json")

# ---------------------
# Caching heavy objects
# ---------------------
@st.cache_resource(show_spinner="Loading model…")
def load_trained_model(model_path: str):
    model = load_model(model_path, compile=False)
    return model

@st.cache_resource(show_spinner="Loading class index mapping…")
def load_class_indices(pth: str) -> Dict:
    with open(pth, "r") as f:
        mapping = json.load(f)
    # Normalize keys: allow both str and int keys in input file
    # Ensure we always return a dict[int -> str]
    fixed = {}
    for k, v in mapping.items():
        try:
            fixed[int(k)] = v
        except:
            fixed[k] = v
    return fixed

# -----------------------
# Utility: image handling
# -----------------------
def get_model_img_size(model) -> Tuple[int, int]:
    """Detect input size from model, fallback to 224x224."""
    try:
        shape = model.input_shape
        h, w = int(shape[1]), int(shape[2])
        if h > 0 and w > 0:
            return (h, w)
    except:
        pass
    return (224, 224)

def read_image_to_rgb(file) -> Image.Image:
    """Load uploaded image file-like -> RGB PIL Image with EXIF orientation fix."""
    img = Image.open(file)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")

def preprocess_for_model(pil_img: Image.Image, target_size: Tuple[int,int]) -> np.ndarray:
    img = pil_img.resize(target_size)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

# -----------------------
# Prediction & Top-K util
# -----------------------
def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)

def predict_topk(model, img_batch: np.ndarray, k: int = 3) -> Tuple[int, float, List[Tuple[int, float]]]:
    preds = model.predict(img_batch, verbose=0)
    if preds.ndim == 1:
        preds = np.expand_dims(preds, 0)
    probs = preds[0]
    # If model not compiled with softmax at end, apply it
    if not np.isclose(probs.sum(), 1.0, atol=1e-3):
        probs = softmax(probs[None, ...])[0]
    top_indices = probs.argsort()[::-1][:k]
    top = [(int(i), float(probs[i])) for i in top_indices]
    return int(top_indices[0]), float(probs[top_indices[0]]), top

# -----------------------
# Grad-CAM implementation
# -----------------------
def find_last_conv_layer(model):
    # Try to find a suitable conv layer near the end
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        # If model has nested models (e.g., Sequential/Functional base)
        if hasattr(layer, "layers"):
            for sub in reversed(layer.layers):
                if isinstance(sub, tf.keras.layers.Conv2D):
                    return sub.name
    return None

def grad_cam(model, img_array, last_conv_layer_name: str = None):
    """Returns heatmap for the given image batch (shape [1, H, W, 3])."""
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)
        if last_conv_layer_name is None:
            return None

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        top_class_channel = predictions[:, pred_index]

    grads = tape.gradient(top_class_channel, conv_outputs)
    if grads is None:
        return None

    # global average pooling over HxW for each channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(conv_outputs, pooled_grads), axis=-1)

    heatmap = np.maximum(heatmap, 0)  # ReLU
    max_val = np.max(heatmap) + 1e-8
    heatmap /= max_val
    return heatmap

def overlay_heatmap(pil_img: Image.Image, heatmap: np.ndarray, alpha: float = 0.35) -> Image.Image:
    """Overlay heatmap on the original PIL image."""
    import cv2
    img = np.array(pil_img)
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_TURBO)
    overlay = cv2.addWeighted(img, 1.0, heatmap_color, alpha, 0)
    return Image.fromarray(overlay)

# -----------------------
# Header / Sidebar
# -----------------------
left, mid, right = st.columns([1, 2, 1])
with mid:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🌿 Plant Disease Classifier")
    st.caption("Upload a leaf image to get the predicted disease along with confidence and Grad-CAM explanation.")
    st.markdown('</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("Adjust preferences below and then classify.")
    st.divider()
    show_gradcam = st.toggle("Show Grad-CAM overlay", value=True, help="Visualize what the model focused on.")
    show_topk = st.toggle("Show top-3 probabilities", value=True)
    k_value = 3
    st.divider()
    st.markdown("### 📦 Model")
    st.write(f"Model path: `{MODEL_PATH}`")
    st.write(f"Classes: `{CLASS_IDX_PATH}`")
    st.info("Make sure the files exist. The app auto-detects input size from the model.")

# -----------------------
# Model & mapping loading
# -----------------------
# Graceful load with messaging
if not os.path.exists(MODEL_PATH):
    st.error(f"Model not found at: {MODEL_PATH}. Please place your .h5 file there.")
    st.stop()

if not os.path.exists(CLASS_IDX_PATH):
    st.error(f"class_indices.json not found at: {CLASS_IDX_PATH}.")
    st.stop()

model = load_trained_model(MODEL_PATH)
class_indices = load_class_indices(CLASS_IDX_PATH)
num_classes = model.output_shape[-1]
img_h, img_w = get_model_img_size(model)

# Safety check if class mapping aligns with model output
if len(class_indices) != num_classes:
    st.warning(
        f"Number of classes in mapping ({len(class_indices)}) doesn't match model output ({num_classes}). "
        "Predictions may be misaligned."
    )

# Reverse mapping if needed
index_to_class = {}
for idx in range(num_classes):
    if idx in class_indices:
        index_to_class[idx] = class_indices[idx]
    else:
        # fallback: string key
        key = str(idx)
        index_to_class[idx] = class_indices.get(key, f"class_{idx}")

# -----------------------
# Uploader — single or multi
# -----------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📤 Upload Image(s)")
uploaded_files = st.file_uploader(
    "Drop leaf images here (JPG/PNG)…",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)
st.markdown('</div>', unsafe_allow_html=True)

if not uploaded_files:
    st.stop()

# -----------------------
# Process & Predict
# -----------------------
for i, upl in enumerate(uploaded_files, start=1):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    cols = st.columns([1, 1.2])
    with cols[0]:
        st.markdown(f"#### 🌱 Image {i}")
        try:
            pil_img = read_image_to_rgb(upl)
        except Exception as e:
            st.error(f"Could not read image: {e}")
            st.markdown('</div>', unsafe_allow_html=True)
            continue
        preview = pil_img.copy().resize((min(420, pil_img.width), int(min(420, pil_img.width) * pil_img.height / pil_img.width)))
        st.image(preview, caption="Uploaded image", use_column_width=True)

        # Download button for the original upload (optional)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        byte_im = buf.getvalue()
        st.download_button("⬇️ Download original", data=byte_im, file_name=f"uploaded_{i}.png", type="secondary")

    with cols[1]:
        st.markdown("#### 🔎 Prediction")
        with st.spinner("Classifying…"):
            batch = preprocess_for_model(pil_img, (img_w, img_h))
            top_idx, top_conf, topk = predict_topk(model, batch, k=3)

        pred_label = index_to_class.get(top_idx, f"class_{top_idx}")

        # Big metric display
        c1, c2 = st.columns([1.2, 1])
        with c1:
            st.metric(label="Predicted class", value=pred_label)
        with c2:
            st.metric(label="Confidence", value=f"{top_conf*100:.1f}%")

        # Show probability bars (top-k)
        if show_topk and topk:
            st.write("**Top-3 probabilities**")
            for idx, p in topk:
                bar = st.progress(min(1.0, max(0.0, p)))
                st.caption(f"{index_to_class.get(idx, f'class_{idx}')} — {p*100:.1f}%")

        # Grad-CAM
        if show_gradcam:
            heatmap = grad_cam(model, batch)
            if heatmap is not None:
                try:
                    overlay = overlay_heatmap(pil_img, heatmap, alpha=0.35)
                    st.image(overlay, caption="Grad-CAM overlay", use_column_width=True)
                    st.caption("Grad-CAM highlights regions the model used for its decision.")
                except Exception as e:
                    st.info(f"Grad-CAM overlay not available: {e}")
            else:
                st.info("Grad-CAM could not locate a convolutional layer in this model.")

        # JSON export of prediction
        result = {
            "predicted_index": int(top_idx),
            "predicted_label": pred_label,
            "confidence": float(top_conf),
            "topk": [(int(i), float(s), index_to_class.get(int(i), f"class_{i}")) for i, s in topk]
        }
        json_bytes = json.dumps(result, indent=2).encode("utf-8")
        st.download_button("📄 Download prediction JSON", data=json_bytes, file_name=f"prediction_{i}.json")

    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------
# Footer note
# -----------------------
st.markdown("""
<div style="text-align:center; color:#6b7280; margin-top: 2rem;">
  <em>Tip:</em> For best results, use sharp, well-lit images of a single leaf against a clean background.
</div>
""", unsafe_allow_html=True)