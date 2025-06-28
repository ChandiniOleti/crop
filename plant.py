import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import io
from utils.model import ResNet9
from utils.disease import disease_dic

# -------------------- Disease Classes --------------------
disease_classes = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# -------------------- Load Model --------------------
@st.cache_resource
def load_model():
    model = ResNet9(3, len(disease_classes))
    model.load_state_dict(torch.load('plant_disease_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# -------------------- Predict Function --------------------
def predict_disease(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)
    yb = model(img_u)
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    return prediction

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="🌿 Plant Disease Detector", layout="wide")

# Sidebar
st.sidebar.title("🌱 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🔍 Predict Disease", "ℹ️ About"])

# -------------------- Pages --------------------
if page == "🏠 Home":
    st.title("🌿 Welcome to Smart Plant Disease Detection")
    st.image("https://images.unsplash.com/photo-1610116306796-798b8cd2213f", use_container_width=True)
    st.markdown("""
    ### 👨‍🌾 Empowering Farmers with AI
    This app detects plant leaf diseases using computer vision and deep learning.
    
    #### ✅ Features:
    - Real-time image analysis
    - AI-generated recommendations
    - Simple and intuitive UI

    Use the sidebar to begin!
    """)

elif page == "🔍 Predict Disease":
    st.title("🔍 Upload Leaf Image for Disease Detection")
    uploaded_file = st.file_uploader("📷 Choose a leaf image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(uploaded_file, caption="Uploaded Image", width=250)

        with col2:
            with st.spinner('🧠 Analyzing...'):
                image_bytes = uploaded_file.read()
                prediction = predict_disease(image_bytes)
                recommendation = disease_dic.get(prediction, "No specific advice available.")

            st.success(f"✅ Prediction: **{prediction}**")
            st.markdown(f"<b>💡 Recommendation:</b><br/><br/>{recommendation}", unsafe_allow_html=True)

elif page == "ℹ️ About":
    st.title("ℹ️ About This App")
    st.markdown("""
    ### 🧠 Plant Disease Detection App  
    Built to help farmers detect plant diseases and take quick action.

    - 🔍 AI Model: ResNet9
    - 📦 Framework: PyTorch + Streamlit
    - 📊 Dataset: PlantVillage
    - 📌 Output: Disease classification + preventive advice

    #### 👩‍💻 Developer:
    **Chandini Oleti**  
    B.Tech in AI & ML | VVIT  
    """)

