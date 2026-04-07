import streamlit as st
import pandas as pd
import numpy as np
import os
import sqlite3

from PIL import Image

# ==============================
# DATABASE SETUP (SQLAlchemy)
# ==============================


    _tablename_ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True)
    password = Column(String)

Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)
db_session = Session()

# ==============================
# OPENAI SETUP
# ==============================
from openai import OpenAI
client = OpenAI(api_key=st.secrets["crop"])

# ==============================
# GEMINI SETUP
# ==============================
import google.generativeai as genai
genai.configure(api_key=st.secrets["yield_key"])
gemini_model = genai.GenerativeModel("gemini-pro")

# ==============================
# LOAD OFFLINE MODEL
# ==============================
try:
    offline_model = joblib.load("crop_model.pkl")
except:
    offline_model = None

# ==============================
# AUTH FUNCTIONS
# ==============================
def register(username, password):
    user = User(username=username, password=password)
    db_session.add(user)
    db_session.commit()

def login(username, password):
    user = db_session.query(User).filter_by(username=username, password=password).first()
    return user

# ==============================
# OFFLINE AI
# ==============================
def offline_response(prompt):
    prompt = prompt.lower()

    if "maize" in prompt:
        return "🌽 Plant maize early with proper spacing (75cm x 25cm)."
    elif "fertilizer" in prompt:
        return "Use basal fertilizer during planting and top dress later."
    elif offline_model:
        try:
            return str(offline_model.predict([prompt])[0])
        except:
            pass

    return "Offline mode: Provide more details."

# ==============================
# OPENAI CHAT
# ==============================
def get_ai_response(prompt):
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return res.choices[0].message.content
    except:
        return offline_response(prompt)

# ==============================
# GEMINI YIELD PREDICTION
# ==============================
def get_yield_prediction(data):
    try:
        prompt = f"Predict crop yield based on: {data}"
        response = gemini_model.generate_content(prompt)
        return response.text
    except:
        return "AI unavailable. Try again later."

# ==============================
# STREAMLIT UI
# ==============================
st.set_page_config(layout="wide")

# ==============================
# AUTH UI
# ==============================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

menu = ["Login", "Register"]

if not st.session_state.logged_in:
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Register":
        st.subheader("Create Account")
        user = st.text_input("Username")
        pwd = st.text_input("Password", type="password")

        if st.button("Register"):
            register(user, pwd)
            st.success("Account created!")

    elif choice == "Login":
        st.subheader("Login")
        user = st.text_input("Username")
        pwd = st.text_input("Password", type="password")

        if st.button("Login"):
            if login(user, pwd):
                st.session_state.logged_in = True
                st.success("Logged in!")
                st.rerun()
            else:
                st.error("Invalid credentials")

# ==============================
# MAIN APP
# ==============================
if st.session_state.logged_in:

    st.title("🌾 AI Crop Advisory System")

    tabs = st.tabs([
        "🌽 Yield Prediction",
        "🐛 Pest Detection",
        "🤖 AI Advisor",
        "📊 Market Trends"
    ])

    # ==========================
    # TAB 1: YIELD
    # ==========================
    with tabs[0]:
        st.header("Crop Yield Prediction")

        crop = st.selectbox("Crop", ["Maize", "Wheat", "Soybeans"])
        rainfall = st.number_input("Rainfall (mm)")
        temp = st.number_input("Temperature (°C)")

        if st.button("Predict Yield"):
            data = {
                "crop": crop,
                "rainfall": rainfall,
                "temperature": temp
            }
            result = get_yield_prediction(data)
            st.success(result)

    # ==========================
    # TAB 2: PEST DETECTION
    # ==========================
    with tabs[1]:
        st.header("Pest Detection")

        file = st.file_uploader("Upload crop image")

        if file:
            img = Image.open(file)
            st.image(img, caption="Uploaded Image")

            st.info("Pest detection model can be integrated here.")

    # ==========================
    # TAB 3: AI ADVISOR
    # ==========================
    with tabs[2]:
        st.header("AI Crop Advisor")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        prompt = st.chat_input("Ask about farming...")

        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.write(prompt)

            response = get_ai_response(prompt)

            st.session_state.messages.append({"role": "assistant", "content": response})

            with st.chat_message("assistant"):
                st.write(response)

    # ==========================
    # TAB 4: MARKET
    # ==========================
    with tabs[3]:
        st.header("Market Trends")

        if os.path.exists("market_trends.csv"):
            df = pd.read_csv("market_trends.csv")
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("No market data available.")

    # ==========================
    # LOGOUT
    # ==========================
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()
