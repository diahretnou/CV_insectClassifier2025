import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import os

# Load models
cnn_model = load_model("ProyekCV_model.h5")
mobilenet_model = load_model("ProyekCV_model_v2.h5")

class_names = ['Butterfly', 'Dragonfly', 'Grasshopper', 'Ladybird', 'Mosquito']

descriptions = {
    'Grasshopper': "Grasshopper adalah serangga herbivora yang dikenal dengan kemampuan melompat jauh berkat kaki belakangnya yang kuat...",
    'Butterfly': "Butterfly adalah serangga cantik dengan sayap berwarna-warni yang hidup di berbagai habitat...",
    'Dragonfly': "Dragonfly adalah serangga pemangsa yang hidup di dekat air...",
    'Ladybird': "Ladybird, atau kepik, adalah serangga kecil berwarna cerah dengan bintik-bintik di punggungnya...",
    'Mosquito': "Mosquito adalah serangga kecil yang dikenal sebagai penghisap darah..."
}

# Image preprocessing
def preprocess_image(image):
    image = image.resize((150, 150))
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

# Streamlit UI
st.set_page_config(page_title="Insect Classifier", layout="centered")
st.title("🦋🪰 Insect Classifier - CNN & MobileNet")
st.caption("Diah Retno Utami - 4TIB")

uploaded_file = st.file_uploader("Upload gambar serangga", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diupload", use_column_width=True)

    img = preprocess_image(image)

    # CNN prediction
    cnn_pred = cnn_model.predict(img)
    cnn_class_idx = int(np.argmax(cnn_pred[0]))
    cnn_conf = float(np.max(cnn_pred[0]))
    cnn_label = class_names[cnn_class_idx] if cnn_conf >= 0.5 else "Tidak Dikenali"
    cnn_desc = descriptions.get(cnn_label, "Gambar tidak dapat dikenali dengan tingkat kepercayaan yang memadai.")

    # MobileNet prediction
    mobile_pred = mobilenet_model.predict(img)
    mobile_class_idx = int(np.argmax(mobile_pred[0]))
    mobile_conf = float(np.max(mobile_pred[0]))
    mobile_label = class_names[mobile_class_idx] if mobile_conf >= 0.5 else "Tidak Dikenali"
    mobile_desc = descriptions.get(mobile_label, "Gambar tidak dapat dikenali dengan tingkat kepercayaan yang memadai.")

    st.subheader("📊 Hasil Prediksi")
    st.markdown(f"**Model CNN**\n- Kelas: `{cnn_label}`\n- Akurasi: `{cnn_conf:.2%}`")
    st.markdown(f"**Model MobileNet**\n- Kelas: `{mobile_label}`\n- Akurasi: `{mobile_conf:.2%}`")

    st.subheader("📖 Deskripsi Serangga")
    st.write(cnn_desc)

