import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import pickle

st.set_page_config(page_title="Sign Language Detection", page_icon="🤟", layout="wide")

st.markdown("""
<style>
.main {background-color: #f5f7fa;}
.stButton>button {
    width: 100%;
    background-color: #4A90E2;
    color: white;
    font-weight: bold;
    border-radius: 10px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("🤟 Sign Language Detection")
st.markdown("### Upload an image to detect sign language gestures")

@st.cache_data
def load_config():
    with open('models/config.pkl', 'rb') as f:
        config = pickle.load(f)
    return config

config = load_config()
IMG_SIZE = config['img_size']
classes = config['classes']

@st.cache_resource
def load_models():
    resnet = load_model('models/resnet50_model.h5')
    mobile = load_model('models/mobilenetv2_model.h5')
    return resnet, mobile

st.sidebar.header("⚙️ Settings")
model_choice = st.sidebar.selectbox("Select Model", ["ResNet50", "MobileNetV2"])
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

resnet_model, mobile_model = load_models()
model = resnet_model if model_choice == "ResNet50" else mobile_model

def preprocess_image(image):
    img = np.array(image)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict(model, image):
    processed_img = preprocess_image(image)
    predictions = model.predict(processed_img, verbose=0)
    predicted_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_idx]
    predicted_class = classes[predicted_idx]
    top_5_idx = np.argsort(predictions[0])[-5:][::-1]
    top_5_classes = [classes[i] for i in top_5_idx]
    top_5_conf = [predictions[0][i] for i in top_5_idx]
    return predicted_class, confidence, top_5_classes, top_5_conf

col1, col2 = st.columns([1, 1])

with col1:
    st.header("📷 Upload Image")
    uploaded_file = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg'])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_container_width=True)

with col2:
    st.header("🎯 Prediction")
    if uploaded_file:
        predicted_class, confidence, top_5_classes, top_5_conf = predict(model, image)
        if confidence >= confidence_threshold:
            st.markdown(f"""
            <div style="background: white; padding: 20px; border-radius: 10px; text-align: center;">
                <h2 style="color: #4A90E2;">Predicted: {predicted_class}</h2>
                <h3>Confidence: {confidence:.2%}</h3>
            </div>
            """, unsafe_allow_html=True)
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence * 100,
                title={'text': "Confidence"},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': "#4A90E2"},
                       'steps': [
                           {'range': [0, 50], 'color': "lightgray"},
                           {'range': [50, 75], 'color': "gray"},
                           {'range': [75, 100], 'color': "lightgreen"}
                       ]}
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("Top 5 Predictions")
            for i, (cls, conf) in enumerate(zip(top_5_classes, top_5_conf)):
                st.write(f"{i+1}. **{cls}**: {conf:.2%}")
        else:
            st.warning(f"Low confidence: {confidence:.2%}")
    else:
        st.info("Upload an image to get started")

st.markdown("---")
st.markdown("Sign Language Detection** | Powered by Deep Learning")