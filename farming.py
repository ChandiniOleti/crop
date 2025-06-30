# smart_farming_app.py

# Completely suppress all warnings
import warnings
import os
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings('ignore')

# Specific suppressions for scikit-learn
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Trying to unpickle estimator.*")
warnings.filterwarnings("ignore", module="sklearn")

import streamlit as st
import pandas as pd
import numpy as np
import base64
import os
import io
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
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

# Define a function to get the absolute path to the images directory
def get_image_path(filename):
    # First try in the static folder (for Streamlit Cloud)
    static_path = os.path.join("static", filename)
    if os.path.exists(static_path):
        return static_path
    
    # Then try relative to current file in images folder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(current_dir, "images", filename)
    if os.path.exists(img_path):
        return img_path
    
    # Last try in images folder
    img_path = os.path.join("images", filename)
    return img_path

CROP_IMAGES = {
    "apple": "apple.jpg", "banana": "banana.jpg", "blackgram": "blackgram.jpg",
    "chickpea": "chickpea.jpg", "coconut": "coconut.jpg", "coffee": "coffee.jpg",
    "cotton": "cotton.jpg", "grapes": "grapes.jpg", "jute": "jute.jpg",
    "kidneybeans": "kidneybeans.jpg", "lentil": "lentil.jpg", "maize": "maize.jpg",
    "mango": "mango.jpg", "mothbeans": "mothbeans.jpg", "mungbean": "mungbean.jpg",
    "muskmelon": "muskmelon.jpg", "orange": "orange.jpg", "papaya": "papaya.jpg",
    "pigeonpeas": "pigeonpeas.jpg", "pomegranate": "pomegranate.jpg",
    "rice": "rice.jpg", "watermelon": "watermelon.jpg"
}

# Fertilizer Setup
CSV_PATH = "Fertilizer_recommendation.csv"
IMG_FOLDER = "images"
FILE_MAP = {
    "Urea": "f1.webp", "DAP": "f2.jpeg", "20-20": "f3.png",
    "10-26-26": "f4.jpg", "14-35-14": "1.jpg", "17-17-17": "2.jpg", "28-28": "fertilizer.jpg",
}
# Using the detailed descriptions from utils/fertilizer_desc.py
from utils.fertilizer_desc import FERTILIZER_DESCRIPTIONS

def norm(t):
    return t.lower().replace(" ", "").replace("-", "").replace("_", "")

def find_image(fert_name):
    folder = Path(IMG_FOLDER)
    
    # For Streamlit Cloud compatibility
    if not folder.exists():
        # Try to create the path relative to the current directory
        current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        folder = current_dir / IMG_FOLDER
        if not folder.exists():
            st.error(f"Image folder {IMG_FOLDER} does not exist!")
            return None

    # 1) dictionary shortcut
    mapped = FILE_MAP.get(fert_name)
    if mapped:
        img_path = folder / mapped
        if img_path.is_file():
            return str(img_path)
        else:
            # Try direct path for Streamlit Cloud
            direct_path = os.path.join(IMG_FOLDER, mapped)
            if os.path.exists(direct_path):
                return direct_path
            st.warning(f"Mapped image {mapped} for {fert_name} not found")

    want = norm(fert_name)
    # 2) exact-stem match
    for p in folder.iterdir():
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp") and norm(p.stem) == want:
            return str(p)
    # 3) substring match
    for p in folder.iterdir():
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp") and want in norm(p.stem):
            return str(p)
    
    # Debug info
    st.info(f"Looking for image for '{fert_name}', normalized as '{want}'")
    
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

/* Custom navigation styling */
div.nav-button {
    padding: 10px 15px;
    border-radius: 10px;
    margin-bottom: 10px;
    cursor: pointer;
    transition: all 0.3s ease;
    border-left: 4px solid transparent;
}

div.nav-button:hover {
    background-color: rgba(15, 118, 110, 0.2);
    transform: translateX(5px);
    border-left: 4px solid #0f766e;
}

div.nav-button.active {
    background-color: rgba(15, 118, 110, 0.3);
    border-left: 4px solid #0f766e;
    font-weight: bold;
    box-shadow: 0 2px 5px rgba(0,0,0,0.2);
}

/* Animation for page transitions */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes slideIn {
    from { opacity: 0; transform: translateX(-30px); }
    to { opacity: 1; transform: translateX(0); }
}

.main .block-container {
    animation: fadeIn 0.5s ease-out;
}

/* Add animations to different elements */
.stMarkdown, .stForm, .stImage, .stButton, .stSelectbox, .stNumberInput {
    animation: slideIn 0.4s ease-out;
}

/* Add a subtle hover effect to buttons */
button {
    transition: all 0.3s ease !important;
}

button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
}
</style>
""", unsafe_allow_html=True)

if "active_page" not in st.session_state:
    st.session_state.active_page = "🏠 Home"

# Add title with animation
st.sidebar.markdown("""
<h2 style='
    text-align: center; 
    color: green; 
    animation: pulse 2s infinite;
'>
    🌱 Smart Farming Assistant
</h2>
<style>
@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.05); }
    100% { transform: scale(1); }
}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)

# Create navigation menu with custom styling
pages = ["🏠 Home", "🌾 Crop Recommendation", "🌿 Plant Disease Detection", 
         "🧪 Fertilizer Recommendation", "🌦️ Weather Forecasting", "ℹ️ About"]

for page in pages:
    # Check if this is the active page
    is_active = st.session_state.active_page == page
    
    # Style based on active state
    if is_active:
        st.sidebar.markdown(f"""
        <div style="
            padding: 10px 15px; 
            border-radius: 10px; 
            background-color: rgba(15, 118, 110, 0.3);
            border-left: 4px solid #0f766e;
            margin-bottom: 10px;
            font-weight: bold;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        ">
            {page}
        </div>
        """, unsafe_allow_html=True)
    else:
        # Create a button with custom styling
        if st.sidebar.button(page, key=f"nav_{page}", 
                            use_container_width=True,
                            help=f"Navigate to {page}"):
            st.session_state.active_page = page
            st.rerun()

page = st.session_state.active_page

# Render selected page only
if page == "🏠 Home":
    # Title and header image
    st.title("👩‍🌾 Welcome to Smart Farming Assistant")
    
    # Add some custom CSS for the page
    st.markdown("""
    <style>
    h1, h2, h3 {
        color: #2e7d32;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Display header image
    image_path = "home.jpeg"
    if os.path.exists(image_path):
        col1, col2, col3 = st.columns([1, 1.5, 1])
        with col2:
            st.image(image_path, width=250)
    else:
        st.warning(f"Image '{image_path}' not found in the directory. Please check the path or filename.")

    # Introduction
    st.markdown("""
    ## 🌾 Smart Farming Assistant: Your AI-Powered Agricultural Companion
    
    Welcome to a comprehensive solution that brings cutting-edge technology to farming practices. 
    Our application combines multiple AI models to provide data-driven recommendations for optimal agricultural outcomes.
    """)
    
    # Feature section using Streamlit columns
    st.markdown("### 🌟 Key Features")
    
    # Create feature cards using Streamlit columns
    col1, col2 = st.columns(2)
    
    # Custom CSS for feature cards with animations
    feature_style = """
    <style>
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); box-shadow: 0 4px 8px rgba(0,0,0,0.2); }
        50% { transform: scale(1.03); box-shadow: 0 6px 12px rgba(0,0,0,0.3); }
        100% { transform: scale(1); box-shadow: 0 4px 8px rgba(0,0,0,0.2); }
    }
    
    .feature-box {
        background: linear-gradient(135deg, #43a047, #1b5e20);
        border-radius: 8px;
        padding: 10px 15px;
        color: white;
        height: 100%;
        margin-bottom: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
        animation: fadeInUp 0.8s ease-out forwards;
    }
    
    .feature-box:hover {
        animation: pulse 2s infinite;
    }
    
    .feature-title {
        text-align: center;
        font-weight: bold;
        border-bottom: 1px solid rgba(255,255,255,0.3);
        padding-bottom: 5px;
        margin-bottom: 10px;
    }
    
    .feature-box ul {
        padding-left: 20px;
        margin-top: 8px;
        font-size: 14px;
    }
    
    .feature-box li {
        margin-bottom: 5px;
    }
    
    .delay-1 { animation-delay: 0.1s; }
    .delay-2 { animation-delay: 0.3s; }
    .delay-3 { animation-delay: 0.5s; }
    .delay-4 { animation-delay: 0.7s; }
    </style>
    """
    
    st.markdown(feature_style, unsafe_allow_html=True)
    
    # First row of features
    with col1:
        st.markdown("""
        <div class="feature-box delay-1">
            <div class="feature-title">🌱 Crop Recommendation</div>
            <ul>
                <li>Input soil parameters (N, P, K, pH)</li>
                <li>Get AI predictions for suitable crops</li>
                <li>View images of recommended crops</li>
                <li>Maximize yield potential</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-box delay-2">
            <div class="feature-title">🍃 Plant Disease Detection</div>
            <ul>
                <li>Upload photos of plant leaves</li>
                <li>Instant diagnosis of 38 diseases</li>
                <li>Detailed treatment advice</li>
                <li>Early detection prevents crop loss</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Second row of features
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("""
        <div class="feature-box delay-3">
            <div class="feature-title">🧪 Fertilizer Recommendation</div>
            <ul>
                <li>Input soil conditions and crop type</li>
                <li>Get personalized suggestions</li>
                <li>Comprehensive descriptions</li>
                <li>Visual references for fertilizers</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="feature-box delay-4">
            <div class="feature-title">🌦️ Weather Forecasting</div>
            <ul>
                <li>Check current weather data</li>
                <li>Plan farming activities</li>
                <li>Make informed decisions</li>
                <li>Optimize planting times</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Benefits section
    st.markdown("### 💡 Benefits of Smart Farming")
    
    # Custom CSS for benefit cards with animations
    benefit_style = """
    <style>
    @keyframes fadeInRight {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes glow {
        0% { box-shadow: 0 0 5px rgba(46, 125, 50, 0.2); }
        50% { box-shadow: 0 0 15px rgba(46, 125, 50, 0.4); }
        100% { box-shadow: 0 0 5px rgba(46, 125, 50, 0.2); }
    }
    
    .benefit-box {
        background-color: #e8f5e9;
        border-radius: 8px;
        padding: 10px 15px;
        border-left: 4px solid #2e7d32;
        margin-bottom: 10px;
        height: 100%;
        box-shadow: 0 3px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        animation: fadeInRight 0.8s ease-out forwards;
    }
    
    .benefit-box:hover {
        transform: translateX(5px);
        border-left-width: 8px;
        animation: glow 2s infinite;
    }
    
    .benefit-title {
        color: #2e7d32;
        font-weight: bold;
        border-bottom: 1px solid #a5d6a7;
        padding-bottom: 5px;
        margin-bottom: 8px;
    }
    
    .benefit-text {
        font-size: 14px;
        color: #333;
    }
    
    .delay-5 { animation-delay: 0.9s; }
    .delay-6 { animation-delay: 1.1s; }
    .delay-7 { animation-delay: 1.3s; }
    .delay-8 { animation-delay: 1.5s; }
    </style>
    """
    
    st.markdown(benefit_style, unsafe_allow_html=True)
    
    # Create benefit cards using Streamlit columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="benefit-box delay-5">
            <div class="benefit-title">📈 Increased Productivity</div>
            <div class="benefit-text">
                Optimize resources and maximize yields through data-driven farming decisions.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="benefit-box delay-6">
            <div class="benefit-title">💰 Cost Reduction</div>
            <div class="benefit-text">
                Apply the right inputs at the right time to minimize waste and expenses.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("""
        <div class="benefit-box delay-7">
            <div class="benefit-title">🌍 Environmental Sustainability</div>
            <div class="benefit-text">
                Reduce chemical usage with targeted applications based on actual needs.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="benefit-box delay-8">
            <div class="benefit-title">🔍 Precision Agriculture</div>
            <div class="benefit-text">
                Make data-driven decisions specific to your farm's unique conditions.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Call to action with animation
    st.markdown("""
    <style>
    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% {transform: translateY(0);}
        40% {transform: translateY(-20px);}
        60% {transform: translateY(-10px);}
    }
    
    .cta-button {
        background: linear-gradient(135deg, #43a047, #1b5e20);
        display: inline-block;
        padding: 15px 30px;
        border-radius: 50px;
        color: white;
        font-weight: bold;
        font-size: 16px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        cursor: pointer;
        animation: bounce 3s infinite;
        animation-delay: 2s;
    }
    </style>
    
    <div style="text-align: center; margin-top: 40px; margin-bottom: 30px;">
        <div class="cta-button">
            Start exploring the features using the navigation menu on the left! 👈
        </div>
    </div>
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
        image_filename = CROP_IMAGES.get(prediction.lower())
        
        if image_filename:
            try:
                # First try to load from static folder (for Streamlit Cloud)
                st.image(f"static/{image_filename}", caption=prediction.capitalize(), width=250)
            except Exception as e1:
                try:
                    # Try with our helper function
                    image_path = get_image_path(image_filename)
                    if os.path.exists(image_path):
                        st.image(Image.open(image_path), caption=prediction.capitalize(), width=250)
                    else:
                        # Try one more approach - direct path
                        try:
                            st.image(f"images/{image_filename}", caption=prediction.capitalize(), width=250)
                        except Exception:
                            # List available images to help debug
                            from pathlib import Path
                            img_dir = Path("static")
                            if img_dir.exists():
                                st.error(f"Image {image_filename} not found. Available images: {[f.name for f in img_dir.glob('*.jpg')[:5]]}...")
                            else:
                                img_dir = Path("images")
                                if img_dir.exists():
                                    st.error(f"Image {image_filename} not found. Available images: {[f.name for f in img_dir.glob('*.jpg')[:5]]}...")
                                else:
                                    st.error("Neither static nor images directory found!")
                except Exception as e2:
                    st.error(f"🚫 Failed to load image: {str(e2)}")
        else:
            st.warning("🚫 No image available for this crop.")

# Disease Detection Page
elif page == "🌿 Plant Disease Detection":
    st.title("🔍 Upload Leaf Image for Disease Detection")
    uploaded_file = st.file_uploader("📷 Upload a leaf image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(uploaded_file, caption="Uploaded Leaf", width=250)
        with col2:
            with st.spinner("🧠 Analyzing..."):
                img_bytes = uploaded_file.read()
                result = predict_disease(img_bytes)
                advice = disease_dic.get(result, "No specific advice available.")
            st.success(f"🤪 Prediction: **{result.replace('_', ' ')}**")
            st.markdown(f"""
                <div style='background-color:#f1f3f4;padding:20px;border-radius:10px;margin-top:20px;'>
                    <h4 style='color:black;'>💡 Recommendation</h4>
                    <div style='color:black;font-size:16px;line-height:1.6;text-align:justify;max-height:400px;overflow-y:auto;'>
                        {advice}
                    </div>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("📸 Please upload a clear leaf image to detect the disease.")

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
        # Removed image display section to avoid issues on Streamlit Cloud
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
    ## 🧠 Smart Farming Assistant
    
    ### 📱 Application Overview
    
    This application is designed to revolutionize farming practices by leveraging artificial intelligence and machine learning technologies. It provides farmers with data-driven insights and recommendations to optimize their agricultural operations.
    
    ---
    
    ### 🛠️ How to Use This App
    
    #### 🌱 Crop Recommendation
    1. Navigate to the **Crop Recommendation** page from the sidebar
    2. Enter your soil parameters (N, P, K values)
    3. Input environmental conditions (temperature, humidity, pH, rainfall)
    4. Optionally enter your city name
    5. Click "Submit" to receive AI-based crop recommendations
    6. View the recommended crop along with its image
    
    #### 🍃 Plant Disease Detection
    1. Go to the **Plant Disease Detection** page
    2. Upload a clear image of the affected plant leaf
    3. Wait for the AI to analyze the image
    4. Review the disease diagnosis
    5. Read the detailed treatment recommendations provided
    
    #### 🧪 Fertilizer Recommendation
    1. Select the **Fertilizer Recommendation** page
    2. Enter environmental parameters (temperature, humidity, moisture)
    3. Select your soil type and crop type
    4. Input your soil's N, P, K values
    5. Submit to receive personalized fertilizer recommendations
    6. View detailed information about the recommended fertilizer
    
    #### 🌦️ Weather Forecasting
    1. Navigate to the **Weather Forecasting** page
    2. Enter your city name
    3. View current weather conditions including temperature, humidity, and more
    
    ---
    
    ### 💻 Technical Details
    
    This application integrates multiple machine learning models:
    
    - **Crop Recommendation**: Random Forest classifier trained on soil and climate data
    - **Disease Detection**: Deep learning model (ResNet9) trained on the PlantVillage dataset
    - **Fertilizer Recommendation**: Decision-based system using soil and crop parameters
    - **Weather Data**: Integration with OpenWeatherMap API for real-time weather information
    
    ---
    
    ### 👥 Team Members:
    1. **Oleti Chandini** (22BQ1A42A5)  
       📧 22BQ1A42A5@vvit.net  
    2. **Kumba Naga Malleswari** (22BQ1A4282)  
       📧 22BQ1A4282@vvit.net  
    3. **K. Jeevan Kumar** (22BQ1A4292)  
       📧 22BQ1A4292@vvit.net  
    4. **N. Vasavi** (22BQ1A42A3)  
       📧 22BQ1A42A3@vvit.net  
    
    ---
    
    ### 🙏 Thank You
    """, unsafe_allow_html=True)
