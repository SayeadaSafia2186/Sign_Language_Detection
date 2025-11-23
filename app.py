import streamlit as st
import numpy as np
import cv2
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import pickle
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Sign Language Detection", page_icon="🤟", layout="wide")

st.markdown("""
<style>
.main {background-color: #f5f7fa;}
.stButton > button {
    width: 100%;
    background-color: #4A90E2;
    color: white;
    font-weight: bold;
    border-radius: 10px;
    padding: 10px;
}
.prediction-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    color: white;
    margin: 10px 0;
}
.resnet-box {
    background: linear-gradient(135deg, #4A90E2 0%, #357ABD 100%);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    color: white;
}
.mobile-box {
    background: linear-gradient(135deg, #50C878 0%, #3CB371 100%);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    color: white;
}
.agreement-yes {
    background: linear-gradient(135deg, #00b09b, #96c93d);
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    color: white;
    font-weight: bold;
}
.agreement-no {
    background: linear-gradient(135deg, #ff416c, #ff4b2b);
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    color: white;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.title("🤟 Sign Language Detection")
st.markdown("### Upload an image to detect sign language gestures using both ResNet50 and MobileNetV2")

@st.cache_data
def load_config():
    try:
        with open('models/config.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        try:
            with open('models/classes.txt', 'r') as f:
                classes = [line.strip() for line in f.readlines()]
            return {'img_size': 128, 'classes': classes, 'num_classes': len(classes)}
        except Exception as e:
            st.error(f"Error loading config: {e}")
            return None

config = load_config()
if config is None:
    st.stop()

IMG_SIZE = config.get('img_size', 128)
classes = config['classes']

@st.cache_resource
def load_models():
    resnet_model = None
    mobile_model = None
    
    resnet_paths = ['models/best_resnet50.keras', 'models/resnet50_model.keras', 'models/resnet50_fixed.keras']
    mobile_paths = ['models/best_mobilenetv2.keras', 'models/mobilenetv2_model.keras', 'models/mobilenetv2_fixed.keras']
    
    for path in resnet_paths:
        if os.path.exists(path):
            try:
                resnet_model = load_model(path, compile=False)
                st.sidebar.success(f"✓ ResNet50 loaded from {path}")
                break
            except Exception as e:
                continue
    
    for path in mobile_paths:
        if os.path.exists(path):
            try:
                mobile_model = load_model(path, compile=False)
                st.sidebar.success(f"✓ MobileNetV2 loaded from {path}")
                break
            except Exception as e:
                continue
    
    if resnet_model is None:
        st.sidebar.error("❌ ResNet50 model not found")
    if mobile_model is None:
        st.sidebar.error("❌ MobileNetV2 model not found")
    
    return resnet_model, mobile_model

st.sidebar.header("⚙️ Settings")
selected_model = st.sidebar.selectbox("Select Model", ["ResNet50", "MobileNetV2"])
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
show_all_classes = st.sidebar.checkbox("Show all class probabilities", False)

st.sidebar.markdown("---")
st.sidebar.header("📊 System Info")
st.sidebar.info(f"TensorFlow: {tf.__version__}")
st.sidebar.info(f"NumPy: {np.__version__}")
st.sidebar.info(f"Classes: {len(classes)}")
st.sidebar.info(f"Image Size: {IMG_SIZE}x{IMG_SIZE}")

resnet_model, mobile_model = load_models()

if resnet_model is None and mobile_model is None:
    st.error("❌ No models loaded. Please check the 'models/' directory.")
    st.stop()

def preprocess_image(image):
    img = np.array(image)
    if len(img.shape) == 2: 
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:  
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=0)

def predict(model, image):
    if model is None:
        return None, 0, [], []
    try:
        processed_img = preprocess_image(image)
        predictions = model.predict(processed_img, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_idx]
        predicted_class = classes[predicted_idx]
        top_5_idx = np.argsort(predictions[0])[-5:][::-1]
        top_5_classes = [classes[i] for i in top_5_idx]
        top_5_conf = [predictions[0][i] for i in top_5_idx]
        return predicted_class, confidence, top_5_classes, top_5_conf, predictions[0]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, 0, [], [], []

col1, col2 = st.columns([1, 2])

with col1:
    st.header("📷 Upload Image")
    uploaded_file = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_column_width=True)

        img_array = np.array(image)
        st.caption(f"Original size: {img_array.shape[1]}x{img_array.shape[0]}")

with col2:
    st.header("🎯 Predictions")
    
    if uploaded_file:
        if selected_model == "ResNet50" and resnet_model is not None:
            with st.spinner("ResNet50 predicting..."):
                pred_class, conf, top5_cls, top5_conf, all_probs = predict(resnet_model, image)
                model_color = "#4A90E2"
                model_name = "ResNet50"
        elif selected_model == "MobileNetV2" and mobile_model is not None:
            with st.spinner("MobileNetV2 predicting..."):
                pred_class, conf, top5_cls, top5_conf, all_probs = predict(mobile_model, image)
                model_color = "#50C878"
                model_name = "MobileNetV2"
        else:
            st.error(f"❌ {selected_model} model not loaded!")
            st.stop()

        color_class = "resnet-box" if selected_model == "ResNet50" else "mobile-box"
        
        st.markdown(f"""
        <div class="{color_class}">
            <h3>{model_name}</h3>
            <h2>{pred_class}</h2>
            <h4>Confidence: {conf:.2%}</h4>
        </div>
        """, unsafe_allow_html=True)

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=conf * 100,
            title={'text': "Confidence %"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': model_color},
                   'steps': [
                       {'range': [0, 50], 'color': "#ffcccc"},
                       {'range': [50, 75], 'color': "#ffffcc"},
                       {'range': [75, 100], 'color': "#ccffcc"}
                   ],
                   'threshold': {'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': confidence_threshold * 100}}
        ))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_column_width=True)

        st.subheader("📊 Top 5 Predictions")
        for i, (cls, c) in enumerate(zip(top5_cls, top5_conf)):
            st.progress(float(c), text=f"{i+1}. {cls}: {c:.2%}")

        fig_top5 = px.bar(
            x=top5_conf, 
            y=top5_cls, 
            orientation='h',
            color_discrete_sequence=[model_color],
            labels={'x': 'Confidence', 'y': 'Class'}
        )
        fig_top5.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_top5, use_column_width=True)

        if show_all_classes:
            st.markdown("---")
            st.subheader("📈 Full Probability Distribution")
            fig_all = px.bar(
                x=classes, 
                y=[p*100 for p in all_probs], 
                color_discrete_sequence=[model_color],
                labels={'x': 'Class', 'y': 'Confidence %'}
            )
            fig_all.update_layout(height=400, xaxis_tickangle=-45, showlegend=False)
            st.plotly_chart(fig_all, use_column_width=True)
    
    else:
        st.info("👆 Upload an image to get started")

        with st.expander("📋 Available Classes"):
            cols = st.columns(7)
            for i, cls in enumerate(classes):
                cols[i % 7].write(f"• {cls}")

st.markdown("---")
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    st.markdown("**🤟 Sign Language Detection**")
with col_f2:
    st.markdown("Powered by Deep Learning")
with col_f3:
    st.markdown(f"Using: {selected_model}")
