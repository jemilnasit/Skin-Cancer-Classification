import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
import xgboost as xgb
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as effnet_preprocess
from tensorflow.keras.applications.convnext import preprocess_input as convnext_preprocess

@st.cache_resource
def load_models():
    convnext = tf.keras.models.load_model("models/convnext-t.keras")
    resnet = tf.keras.models.load_model("models/resnet50.keras")
    effnet = tf.keras.models.load_model("models/effnetv2-s.keras")

    meta_model = xgb.XGBClassifier()
    meta_model.load_model("models/xgb_meta.json")
    return convnext, resnet, effnet, meta_model

convnext, resnet, effnet, meta_model = load_models()

def preprocess_convnext(image, target_size=(224, 224)):
    img_array = np.array(image)
    h, w, _ = img_array.shape
    min_side = min(h, w)
    top = (h - min_side) // 2
    left = (w - min_side) // 2
    img_array = img_array[top:top + min_side, left:left + min_side]
    img_array = tf.image.resize(img_array, target_size)
    img_array = convnext_preprocess(img_array)
    return np.expand_dims(img_array, axis=0)

def preprocess_efficientnet(image, target_size=(300, 300)):
    img_array = np.array(image)
    h, w, _ = img_array.shape
    min_side = min(h, w)
    top = (h - min_side) // 2
    left = (w - min_side) // 2
    img_array = img_array[top:top + min_side, left:left + min_side]
    img_array = tf.image.resize(img_array, target_size)
    img_array = effnet_preprocess(img_array)
    return np.expand_dims(img_array, axis=0)

def preprocess_resnet(image, target_size=(224, 224)):
    img_array = np.array(image)
    h, w, _ = img_array.shape
    min_side = min(h, w)
    top = (h - min_side) // 2
    left = (w - min_side) // 2
    img_array = img_array[top:top + min_side, left:left + min_side]
    img_array = tf.image.resize(img_array, target_size)
    img_array = resnet_preprocess(img_array)
    return np.expand_dims(img_array, axis=0)

def ensemble_predict(image):
    img_convnext = preprocess_convnext(image)
    img_efficient = preprocess_efficientnet(image)
    img_resnet = preprocess_resnet(image)

    pred1 = convnext.predict(img_convnext, verbose=1)
    pred2 = effnet.predict(img_efficient, verbose=1)
    pred3 = resnet.predict(img_resnet, verbose=1)

    stacked = np.concatenate([pred1, pred2, pred3], axis=1)
    final_pred = meta_model.predict(stacked)
    final_proba = meta_model.predict_proba(stacked)
    confidence = final_proba[0, final_pred[0]]

    return final_pred[0], confidence

DISEASE_INFO = {
    "AKIEC": {
        "name": "Actinic Keratoses and Intraepithelial Carcinoma",
        "description": "A precancerous area of thick, scaly, or crusty skin that often feels dry or rough. Usually caused by long-term exposure to sunlight.",
        "advice": "Avoid sun exposure, use sunscreen, and visit a dermatologist for possible cryotherapy or topical treatment."
    },
    "BCC": {
        "name": "Basal Cell Carcinoma",
        "description": "A common type of skin cancer that often appears as a slightly transparent bump on the skin, though it can take other forms.",
        "advice": "Consult a dermatologist promptly. Early treatment usually leads to full recovery."
    },
    "BKL": {
        "name": "Benign Keratosis-like Lesion",
        "description": "A non-cancerous growth that often looks like a wart or mole. It may darken or grow over time.",
        "advice": "Generally harmless, but check for irregular borders or rapid growth."
    },
    "DF": {
        "name": "Dermatofibroma",
        "description": "A benign skin nodule that feels firm and is usually found on the legs. Often caused by minor skin injury.",
        "advice": "No treatment required unless it becomes painful or enlarges; consult a dermatologist if it changes."
    },
    "MEL": {
        "name": "Melanoma",
        "description": "A serious form of skin cancer that begins in pigment-producing cells (melanocytes). Early detection is critical.",
        "advice": "Urgent dermatology consultation is necessary. Monitor for asymmetry, color variation, and irregular borders."
    },
    "NV": {
        "name": "Melanocytic Nevus (Mole)",
        "description": "A common mole or skin growth formed by clusters of pigment cells.",
        "advice": "Usually harmless, but monitor for sudden changes in shape or color."
    },
    "VASC": {
        "name": "Vascular Lesion",
        "description": "Includes angiomas and other benign growths involving blood vessels.",
        "advice": "Usually non-cancerous. Laser or minor surgery may be used for cosmetic removal."
    }
}

st.title("🩺 Skin Cancer Detection")

uploaded_file = st.file_uploader("📸 Upload an image", type=["jpg", "jpeg", "png"])

CLASS_NAMES = ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Predict Disease"):
        with st.spinner("Analyzing image..."):
            pred_label, confidence = ensemble_predict(image)
        confidence = confidence * 100
        
        class_key = CLASS_NAMES[int(pred_label)]
        info = DISEASE_INFO.get(class_key, {})

        st.success(f"✅ **Predicted Class:**{class_key}: {info.get('name', class_key)}")
        st.info(f"📊 **Confidence:** {confidence:.2f}")

        st.subheader("🩻 Disease Information")
        st.write(f"**Description:** {info.get('description', 'No information available.')}")
        st.write(f"**Advice:** {info.get('advice', 'Please consult a dermatologist for professional guidance.')}")
        
        if confidence < 70:
            st.warning("⚠️ The confidence level is low. It is recommended to consult a dermatologist for an accurate diagnosis.")


st.caption("☠️ Disclaimer: This tool is for educational and research purposes only. It is not a substitute for professional medical advice.")
