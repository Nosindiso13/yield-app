import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import io
from PIL import Image
import asyncio


# Imports for Gemini API
import google.generativeai as genai

# --- Configuration --- 
MODEL_PATH = 'model_artifacts/xgboost_pipeline.joblib'
MARKET_PATH = 'market_trends.csv'


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

# In-memory user store for demonstration. 
# In a real application, this would be a database.
# {username: {"password": hashed_password, "role": "user"}}


       

# --- Caching Models for Efficiency ---
@st.cache_resource
def load_models():
    yield_model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
    pest_model = None # No actual model needed for mock detection
    api_key = os.getenv('YIELD_API_KEY')
    if not api_key:
        st.error("Google API Key (YIELD_API_KEY) not found. Please set it as an environment variable.")
        return yield_model, pest_model, None
    genai.configure(api_key=api_key)
    gemini = genai.GenerativeModel('gemini-pro')
    return yield_model, pest_model, gemini

# Helper function for mock pest detection
def detect_pest_mock(image_bytes: bytes) -> list[dict]:
    if image_bytes:
        return [
            {"label": "mock_pest_1", "description": "Aphids", "probability": 0.95},
            {"label": "mock_pest_2", "description": "Leaf Blight", "probability": 0.80}
        ]
    return []

yield_pipeline, pest_model, gemini_model = load_models()

st.set_page_config(page_title='Farmer Advisor & Market', layout='wide')
st.title('🌾 Crop Advisor & Marketplace')

# --- Authentication Logic ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.user_role = None

def login():
    st.subheader("Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            if username in st.session_state.users:
                user_data = st.session_state.users[username]
                if verify_password(password, user_data["password"]):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.user_role = user_data["role"]
                    st.success(f"Welcome, {username}!")
                    st.experimental_rerun() # Rerun to show authenticated content
                else:
                    st.error("Incorrect password.")
            else:
                st.error("Username not found.")

def register():
    st.subheader("Register New User")
    with st.form("register_form"):
        new_username = st.text_input("New Username")
        new_password = st.text_input("New Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submitted = st.form_submit_button("Register")

        if submitted:
            if new_username in st.session_state.users:
                st.error("Username already exists.")
            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            elif len(new_password) < 6:
                st.error("Password must be at least 6 characters long.")
            else:
                st.session_state.users[new_username] = {"password": hash_password(new_password), "role": "user"}
                st.success("Registration successful! Please login.")
                st.session_state.logged_in = False # Force login after registration
                st.experimental_rerun()

if not st.session_state.logged_in:
    st.sidebar.title("Authentication")
    auth_option = st.sidebar.radio("", ["Login", "Register"])
    if auth_option == "Login":
        login()
    else:
        register()

    st.stop() # Stop execution if not logged in

# --- Main App Content (only shown if logged in) ---
st.sidebar.write(f"Logged in as: {st.session_state.username} ({st.session_state.user_role})")
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.user_role = None
    st.experimental_rerun()


tabs = st.tabs(['📈 Yield Prediction', '🪲 Pest Detection', '🤖 AI Advisor', '🛒 Market & Trends'])

with tabs[0]:
    st.header('Yield Prediction')
    with st.form('yield_form'):
        area = st.selectbox('Area', ['Zambia', 'Zimbabwe'])
        item = st.selectbox('Crop', ['Wheat', 'Maize', 'Potatoes', 'Rice, paddy', 'Sorghum', 'Soybeans'])
        year = st.number_input('Year', 2024, 2030, 2025)
        rain = st.slider('Rainfall (mm)', 0, 3500, 1000)
        pest = st.slider('Pesticides (tonnes)', 0, 150000, 5000)
        temp = st.slider('Temp (°C)', 10, 45, 25)
        if st.form_submit_button('Predict'):
            if yield_pipeline:
                input_df = pd.DataFrame([[area, item, year, rain, pest, temp]], columns=['Area', 'Item', 'Year', 'rainfall', 'pesticides', 'temperature'])
                try:
                    pred = np.expm1(yield_pipeline.predict(input_df))[0]
                    st.success(f'Estimated Yield: {pred:,.2f} hg/ha')
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
            else:
                st.error("Crop Yield Model not loaded. Cannot make prediction.")

with tabs[1]:
    st.header('Pest Identification')
    file = st.file_uploader('Upload leaf or pest image', type=['jpg', 'png'])
    if file:
        img = Image.open(io.BytesIO(file.getvalue())).resize((224, 224))
        st.image(img)
        st.write('Performing mock detection...')
        detections = detect_pest_mock(file.getvalue())
        if detections:
            st.success("Mock Detection Results:")
            for detection in detections:
                st.write(f"- **{detection['description']}** (Confidence: {detection['probability']:.2f})")
        else:
            st.info("No detections found.")

with tabs[2]:
    st.header('AI Chatbot')
    if 'messages' not in st.session_state: st.session_state.messages = []
    for m in st.session_state.messages: st.chat_message(m['role']).write(m['content'])
    if prompt := st.chat_input('How can I improve my soil?'):
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        st.chat_message('user').write(prompt)
        if gemini_model:
            response = gemini_model.generate_content(prompt)
            st.session_state.messages.append({'role': 'assistant', 'content': response.text})
            st.chat_message('assistant').write(response.text)
        else:
            st.error("AI Chatbot not available. Please check API key configuration.")

with tabs[3]:
    st.header('🛒 Farmer Marketplace & Trending Crops')
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Trending Crops This Season')
        if os.path.exists(MARKET_PATH):
            df_trends = pd.read_csv(MARKET_PATH)
            st.dataframe(df_trends, use_container_width=True)
        else: st.info('No trend data available.')
    with col2:
        st.subheader('List Your Crop for Sale')
        with st.form('market_form'):
            seller_name = st.text_input('Name')
            crop_type = st.selectbox('Crop', ['Wheat', 'Maize', 'Potatoes', 'Rice, paddy', 'Sorghum', 'Soybeans'])
            quantity = st.number_input('Quantity (kg)', min_value=1)
            price = st.number_input('Asking Price ($)', min_value=1)
            if st.form_submit_button('Post Listing'):
                st.success(f'Listing created for {seller_name}! Others can now see your {crop_type}.')
