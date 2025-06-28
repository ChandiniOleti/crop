# smart_farming_app.py

import streamlit as st
import pandas as pd
import numpy as np
import base64
import os
import io
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
import pickle
import requests

from utils.model import ResNet9
from utils.disease import disease_dic
from utils.fertilizer_desc import FERTILIZER_DESCRIPTIONS

# Disease Classes
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

@st.cache_resource
def load_disease_model():
    model = ResNet9(3, len(disease_classes))
    model.load_state_dict(torch.load('plant_disease_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

disease_model = load_disease_model()

def predict_disease(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)
    yb = disease_model(img_u)
    _, preds = torch.max(yb, dim=1)
    return disease_classes[preds[0].item()]

# Load Crop Recommender Model
with open("crop_recommender.pkl", "rb") as file:
    crop_model = pickle.load(file)

CROP_IMAGES = {
    "apple": "Images/apple.jpg", "banana": "Images/banana.jpg", "blackgram": "Images/blackgram.jpg",
    "chickpea": "Images/chickpea.jpg", "coconut": "Images/coconut.jpg", "coffee": "Images/coffee.jpg",
    "cotton": "Images/cotton.jpg", "grapes": "Images/grapes.jpg", "jute": "Images/jute.jpg",
    "kidneybeans": "Images/kidneybeans.jpg", "lentil": "Images/lentil.jpg", "maize": "Images/maize.jpg",
    "mango": "Images/mango.jpg", "mothbeans": "Images/mothbeans.jpg", "mungbean": "Images/mungbean.jpg",
    "muskmelon": "Images/muskmelon.jpg", "orange": "Images/orange.jpg", "papaya": "Images/papaya.jpg",
    "pigeonpeas": "Images/pigeonpeas.jpg", "pomegranate": "Images/pomegranate.jpg",
    "rice": "Images/rice.jpg", "watermelon": "Images/watermelon.jpg"
}

# Fertilizer Setup
CSV_PATH = "Fertilizer_recommendation.csv"
IMG_FOLDER = "images"
FILE_MAP = {
    "Urea": "f1.webp", "DAP": "f2.jpeg", "20-20": "f3.png",
    "10-26-26": "f4.jpg", "14-35-14": "1.jpg", "17-17-17": "2.jpg", "28-28": "fertilizer.jpg",
}
FERT_DESC = {
    "Urea": "Supplies **46 % Nitrogen**. Broadcast close to the root zone.",
    "DAP": "18-46-0. Boosts early root growth; keep seed 5 cm away.",
    "20-20": "Balanced N-P for vegetative stage; add K if soil test is low.",
}

def norm(t):
    return t.lower().replace(" ", "").replace("-", "").replace("_", "")

def find_image(fert_name):
    folder = Path(IMG_FOLDER)
    if not folder.exists():
        return None
    mapped = FILE_MAP.get(fert_name)
    if mapped and (folder / mapped).is_file():
        return str(folder / mapped)
    want = norm(fert_name)
    for p in folder.iterdir():
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp") and norm(p.stem) == want:
            return str(p)
    for p in folder.iterdir():
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp") and want in norm(p.stem):
            return str(p)
    return None

API_KEY = "1a87e7efe6b61b3c5eb08d8042345272"

# Set page config only once and define navigation
st.set_page_config(page_title="Smart Farming Assistant", layout="wide")
st.markdown("""
<style>
.sidebar .sidebar-content { background-color: #111827; color: white; }
.sidebar .block-container { padding: 20px; }
.button-style {
    background-color: #0f766e; color: white; border-radius: 10px;
    padding: 10px; font-weight: bold; margin-bottom: 8px; text-align: center;
}
</style>
""", unsafe_allow_html=True)

if "active_page" not in st.session_state:
    st.session_state.active_page = "🏠 Home"

def nav_button(label, target):
    if st.sidebar.button(label):
        st.session_state.active_page = target

st.sidebar.markdown("<h2 style='text-align: center; color: green;'>🌱 Smart Farming Assistant</h2>", unsafe_allow_html=True)
nav_button("🏠 Home", "🏠 Home")
nav_button("🌾 Crop Recommendation", "🌾 Crop Recommendation")
nav_button("🌿 Plant Disease Detection", "🌿 Plant Disease Detection")
nav_button("🧪 Fertilizer Recommendation", "🧪 Fertilizer Recommendation")
nav_button("🌦️ Weather Forecasting", "🌦️ Weather Forecasting")
nav_button("ℹ️ About", "ℹ️ About")

page = st.session_state.active_page

# Render selected page only
if page == "🏠 Home":
    st.title("👩‍🌾 Welcome to Smart Farming Assistant")
    image_path = "home.jpeg"
    if os.path.exists(image_path):
        st.markdown(f"""
            <div style='display: flex; justify-content: center;'>
                <img src="data:image/jpeg;base64,{base64.b64encode(open(image_path, 'rb').read()).decode()}" width="400" style='border-radius: 10px;'/>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.warning(f"Image '{image_path}' not found in the directory. Please check the path or filename.")

    st.markdown("""
        ### 🌾 Predict Best Crops | 🍃 Detect Leaf Diseases | 🧪 Get Fertilizer Advice  
        Built for farmers to get instant AI-based recommendations.

        ----

        #### 🌱 Why use this app?
        - 🚀 **Fast and Accurate** predictions using Machine Learning
        - 💼 **Supports agricultural decisions** for better productivity
        - 🤝 **Easy-to-use** and mobile-friendly interface

        #### ✨ Try out features like:
        - 📊 Crop suggestion based on soil and weather
        - 📷 Uploading a leaf image to detect disease
        - 🧪 Get smart fertilizer advice with visuals
        - 🌦️ See your city’s current weather
        - 📱 Designed to be useful for farmers, students, and agri-enthusiasts
    """, unsafe_allow_html=True)


elif page == "🌾 Crop Recommendation":
    st.title("🌾 Predict Suitable Crop")
    with st.form("crop_form"):
        N = st.number_input("Nitrogen (N)", 0, 200)
        P = st.number_input("Phosphorous (P)", 0, 200)
        K = st.number_input("Potassium (K)", 0, 200)
        temp = st.number_input("Temperature (°C)", 0.0, 50.0)
        humidity = st.number_input("Humidity (%)", 0.0, 100.0)
        ph = st.number_input("Soil pH", 0.0, 14.0)
        rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0)
        city = st.text_input("City (optional)")
        submitted = st.form_submit_button("Submit")

    if submitted:
        input_data = pd.DataFrame([[N, P, K, temp, humidity, ph, rainfall]],
                                  columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])
        prediction = crop_model.predict(input_data)[0]
        st.success(f"✅ Recommended Crop for {city or 'your region'} is: **{prediction.capitalize()}**")
        image_path = CROP_IMAGES.get(prediction.lower())
        if image_path and os.path.exists(image_path):
            st.image(Image.open(image_path), caption=prediction.capitalize(), width=250)
        else:
            st.warning("🚫 Image not available.")

elif page == "🌿 Plant Disease Detection":
    st.title("🔍 Upload Leaf Image for Disease Detection")
    uploaded_file = st.file_uploader("📷 upload image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(uploaded_file, caption="home.jpeg", width=250)
        with col2:
            with st.spinner("🧠 Analyzing..."):
                img_bytes = uploaded_file.read()
                result = predict_disease(img_bytes)
                advice = disease_dic.get(result, "No specific advice available.")
            st.success(f"🧪 Prediction: {result}")
            st.markdown(f"""
                <div style='background-color:#f1f3f4;padding:20px;border-radius:10px;margin-top:20px;'>
                    <h4>💡 Recommendation</h4>
                    <div style='color:#333;font-size:16px;line-height:1.6;text-align:justify;max-height:400px;overflow-y:auto;'>
                        {advice}
                    </div>
                </div>
            """, unsafe_allow_html=True)

elif page == "🧪 Fertilizer Recommendation":
    st.title("🧪 Smart Fertilizer Recommendation")
    try:
        df = pd.read_csv(CSV_PATH)
        df.columns = df.columns.str.strip()
    except Exception as e:
        st.error(f"❌ Cannot read `{CSV_PATH}` – {e}")
        st.stop()

    soil_opts = sorted(df["Soil Type"].unique())
    crop_opts = sorted(df["Crop Type"].unique())

    with st.form("fert_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            temp = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
            humidity = st.number_input("Humidity (%)", 0.0, 100.0, 60.0)
            moisture = st.number_input("Soil Moisture (%)", 0.0, 100.0, 30.0)
        with c2:
            soil = st.selectbox("Soil Type", soil_opts)
            crop = st.selectbox("Crop Type", crop_opts)
        with c3:
            N = st.number_input("Nitrogen (N)", 0, 300, 50)
            P = st.number_input("Phosphorous (P)", 0, 300, 40)
            K = st.number_input("Potassium (K)", 0, 300, 50)
        go = st.form_submit_button("Submit")

    if go:
        subset = df[(df["Soil Type"] == soil) & (df["Crop Type"] == crop)].copy()
        subset = subset if not subset.empty else df.copy()
        user_vec = np.array([temp, humidity, moisture, N, K, P], float)
        subset["dist"] = np.linalg.norm(
            subset[["Temparature", "Humidity", "Moisture", "Nitrogen", "Potassium", "Phosphorous"]].values - user_vec,
            axis=1)
        fert = subset.sort_values("dist").iloc[0]["Fertilizer Name"]
        st.success(f"### Recommended Fertilizer → **{fert}**")
        img_path = find_image(fert)
        if img_path:
            with open(img_path, "rb") as f:
                enc = base64.b64encode(f.read()).decode()
            st.markdown(
                f"<p style='text-align:center'>"
                f"<img src='data:image/jpeg;base64,{enc}' width='260' style='border-radius:8px;border:2px solid #333'/></p>",
                unsafe_allow_html=True)
        else:
            st.warning(f"🚫 No image found for **{fert}**. Add one to the *{IMG_FOLDER}/* folder.")
        st.markdown("#### Description")
        st.markdown(FERTILIZER_DESCRIPTIONS.get(fert, "_No description available._"), unsafe_allow_html=True)


elif page == "🌦️ Weather Forecasting":
    st.title("🌦️ Weather Forecast")
    city = st.text_input("🔍 Enter your city", "Guntur")
    if st.button("Get Weather"):
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            temp = data["main"]["temp"]
            humidity = data["main"]["humidity"]
            wind = data["wind"]["speed"]
            desc = data["weather"][0]["description"].capitalize()

            st.markdown(f"""
            <div style='background:linear-gradient(to right,#c6ffdd,#fbd786,#f7797d);padding:30px;border-radius:15px;box-shadow:0 4px 12px rgba(0,0,0,0.1);text-align:center;border:3px solid #0288d1;'>
                <h2 style='color:#004d40'>{city.title()}</h2>
                <img src="https://openweathermap.org/img/wn/{data['weather'][0]['icon']}@2x.png">
                <h1 style='font-size:48px;color:#37474f'>{temp}°C</h1>
                <p><b style='font-size:20px;color:#bf360c'>{desc}</b></p>
                <p style='font-size:16px;color:#1b1b1b'>💧 <b>{humidity}%</b> Humidity &nbsp;&nbsp;&nbsp; 🌬️ <b>{wind} Km/h</b> Wind</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("❌ Unable to fetch weather data. Please check your city name.")

elif page == "ℹ️ About":
    st.title("ℹ️ About This App")
    st.markdown("""
    ### 🧠 Smart Farming Assistant  
    Combines AI models for:
    - 🌾 Crop Recommendation
    - 🍃 Plant Disease Detection
    - 🧪 Fertilizer Suggestion
    - 🌦️ Weather Forecasting

    #### 👥 Team Members:
    1. **Oleti Chandini** (22BQ1A42A5)  
       📧 22BQ1A42A5@vvit.net  
    2. **Kumba Naga Malleswari** (22BQ1A4282)  
       📧 22BQ1A4282@vvit.net  
    3. **K. Jeevan Kumar** (22BQ1A4292)  
       📧 22BQ1A4292@vvit.net  
    4. **N. Vasavi** (22BQ1A42A3)  
       📧 22BQ1A42A3@vvit.net  
    """)
