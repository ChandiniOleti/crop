# weather_tab.py
import streamlit as st
import requests

API_KEY = "1a87e7efe6b61b3c5eb08d8042345272"

st.set_page_config(page_title="Weather Forecast", page_icon="🌦️", layout="centered")
st.title("🌦️ Weather Forecast")

st.markdown("""
<style>
html,body,[class*="css"]{
    background: linear-gradient(to right, #e0f7fa, #fff3e0);
    color:#1F1F1F;
    font-family: 'Segoe UI', sans-serif;
}
.weather-card {
    background: #ffffff;
    padding: 30px;
    border-radius: 15px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    max-width: 400px;
    margin: auto;
    text-align: center;
    border: 3px solid #0288d1;
}
.weather-card h1 {
    font-size: 48px;
    color: #00796b;
}
.weather-card h2 {
    font-size: 28px;
    color: #d84315;
    margin-bottom: 10px;
}
.weather-card p {
    font-size: 16px;
    color: #424242;
    margin: 4px 0;
}
</style>
""", unsafe_allow_html=True)

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
        <div class="weather-card">
            <h2>{city.title()}</h2>
            <img src="https://openweathermap.org/img/wn/{data['weather'][0]['icon']}@2x.png">
            <h1>{temp}°C</h1>
            <p><b>{desc}</b></p>
            <p>💧 <b>{humidity}%</b> Humidity &nbsp;&nbsp;&nbsp; 🌬️ <b>{wind} Km/h</b> Wind</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("❌ Unable to fetch weather data. Please check your city name.")
