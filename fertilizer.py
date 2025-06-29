# fertilizer_app.py
# ──────────────────────────────────────────────────────────
# Fertiliser Recommendation – image-robust version
# • CSV similarity logic (no sklearn pickle)
# • Explicit FILE_MAP + intelligent fallback search
# ──────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import base64, os
from pathlib import Path
from utils.fertilizer_desc import FERTILIZER_DESCRIPTIONS

# ─────── PATHS ────────────────────────────────────────────
CSV_PATH   = "Fertilizer_recommendation.csv"
IMG_FOLDER = "images"          # folder that contains 1.jpg, f1.webp, core.jpg …

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

# ─────── 1.  EDIT THIS ONCE  –  map fertiliser → filename ─
FILE_MAP = {
    # 'CSV-name' : 'your_image_file'
    "Urea"      : "f1.webp",
    "DAP"       : "f2.jpeg",
    "20-20"     : "f3.png",
    "10-26-26"  : "f4.jpg",
    "14-35-14"  : "1.jpg",
    "17-17-17"  : "2.jpg",
    "28-28"     : "fertilizer.jpg",
}
# ─────── 2. Using detailed descriptions from utils/fertilizer_desc.py ───────
# FERTILIZER_DESCRIPTIONS is imported from utils.fertilizer_desc

# ─────── UI setup ─────────────────────────────────────────
st.set_page_config(page_title="Fertilizer Recommender",
                   page_icon="🧪",
                   layout="centered")
st.title("🧪 Smart Fertilizer Recommendation")

# dark-mode tweak
st.markdown("""
<style>
html,body,[class*="css"]{background:#0E1117;color:#F0F0F0;}
.stButton>button{background:#F97316;color:white;border:none;border-radius:6px;}
code,pre{background:#1E2228;}
</style>""", unsafe_allow_html=True)

# ─────── load dataset ─────────────────────────────────────
try:
    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip()   # remove stray spaces
except Exception as e:
    st.error(f"❌ Cannot read `{CSV_PATH}` – {e}")
    st.stop()

soil_opts = sorted(df["Soil Type"].unique())
crop_opts = sorted(df["Crop Type"].unique())

# ─────── helper to find an image robustly ────────────────
ALLOWED = (".jpg", ".jpeg", ".png", ".webp")
def norm(t: str) -> str:      # simplify for comparison
    return t.lower().replace(" ", "").replace("-", "").replace("_", "")

def find_image(fert_name: str) -> str | None:
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
        # Try multiple paths for Streamlit Cloud compatibility
        # First try the standard path
        img_path = folder / mapped
        if img_path.is_file():
            return str(img_path)
        
        # Try direct path
        direct_path = os.path.join(IMG_FOLDER, mapped)
        if os.path.exists(direct_path):
            return direct_path
            
        # If we get here, the mapped file wasn't found
        st.warning(f"Mapped image {mapped} for {fert_name} not found")

    want = norm(fert_name)
    # 2) exact-stem match
    for p in folder.iterdir():
        if p.suffix.lower() in ALLOWED and norm(p.stem) == want:
            return str(p)
    # 3) substring match
    for p in folder.iterdir():
        if p.suffix.lower() in ALLOWED and want in norm(p.stem):
            return str(p)
    
    # Debug info
    st.info(f"Looking for image for '{fert_name}', normalized as '{want}'")
    
    return None

# ─────── input form ───────────────────────────────────────
with st.form("fert_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        temp      = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
        humidity  = st.number_input("Humidity (%)",     0.0, 100.0, 60.0)
        moisture  = st.number_input("Soil Moisture (%)",0.0, 100.0, 30.0)
    with c2:
        soil      = st.selectbox("Soil Type", soil_opts)
        crop      = st.selectbox("Crop Type", crop_opts)
    with c3:
        N = st.number_input("Nitrogen (N)",     0, 300, 50)
        P = st.number_input("Phosphorous (P)",  0, 300, 40)
        K = st.number_input("Potassium (K)",    0, 300, 50)
    go = st.form_submit_button("Submit")

# ─────── recommendation logic ─────────────────────────────
if go:
    subset = df[(df["Soil Type"] == soil) & (df["Crop Type"] == crop)].copy()
    subset = subset if not subset.empty else df.copy()

    user_vec = np.array([temp, humidity, moisture, N, K, P], float)
    subset["dist"] = np.linalg.norm(
        subset[["Temparature", "Humidity", "Moisture",
                "Nitrogen", "Potassium", "Phosphorous"]].values - user_vec,
        axis=1)
    fert = subset.sort_values("dist").iloc[0]["Fertilizer Name"]

    st.success(f"### Recommended Fertilizer → **{fert}**")

    # ─── show picture
    # Get the mapped filename for this fertilizer
    mapped_filename = FILE_MAP.get(fert)
    
    if mapped_filename:
        # First try to load from static folder (for Streamlit Cloud)
        try:
            st.image(f"static/{mapped_filename}", width=260, caption=fert)
        except Exception as e1:
            # If that fails, try with our helper function
            try:
                img_path = get_image_path(mapped_filename)
                if os.path.exists(img_path):
                    # Try to open and display the image using base64 encoding
                    with open(img_path, "rb") as f:
                        enc = base64.b64encode(f.read()).decode()
                    st.markdown(
                        f"<p style='text-align:center'>"
                        f"<img src='data:image/jpeg;base64,{enc}' width='260'"
                        f" style='border-radius:8px;border:2px solid #333'/></p>",
                        unsafe_allow_html=True)
                else:
                    # Try one more approach - direct path
                    try:
                        st.image(f"images/{mapped_filename}", width=260, caption=fert)
                    except Exception:
                        # List available images to help debug
                        img_dir = Path("static")
                        if img_dir.exists():
                            st.error(f"Image {mapped_filename} not found in static. Available images: {[f.name for f in img_dir.glob('*.*')[:5]]}...")
                        else:
                            img_dir = Path("images")
                            if img_dir.exists():
                                st.error(f"Image {mapped_filename} not found in images. Available images: {[f.name for f in img_dir.glob('*.*')[:5]]}...")
                            else:
                                st.error("Neither static nor images directory found!")
            except Exception as e2:
                st.error(f"🚫 Failed to load image: {str(e2)}")
    else:
        st.warning(f"🚫 No image mapping found for fertilizer: {fert}")

    # ─── description
    st.markdown("#### Description")
    st.markdown(FERTILIZER_DESCRIPTIONS.get(fert, "_No description yet – add it in `utils/fertilizer_desc.py`._"),
                unsafe_allow_html=True)
