import streamlit as st
import pandas as pd
import numpy as np
import os
import io
from PIL import Image
import asyncio
from datetime import datetime, timedelta
#AI APIs





# ---------------- CONFIG ----------------
MODEL_PATH = 'model_artifacts/xgboost_pipeline.joblib'



# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_crop_yield_model():
    if not os.path.exists(MODEL_PATH):
    
        return None
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_pest_model():
    from tensorflow.keras.applications import MobileNetV2
    return MobileNetV2(weights="imagenet")

@st.cache_resource
def load_openai_client():
    try:
        api_key = userdata.get("crops") # Assuming a secret named 'OPENAI_API_KEY'
        if not api_key:
            st.warning("OpenAI API Key not found. Please set 'OPENAI_API_KEY' in Colab secrets to use OpenAI chat.")
            return None
        client = openai.OpenAI(api_key="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")
        return client
    except Exception as e:
        st.error(f"Error initializing OpenAI client: {e}")
        return None


# --- Authentication Logic (Direct DB Interaction) ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None

def login_form():
    st.subheader("Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            db = next(get_db())
            user_data = db.query(DBUser).filter(DBUser.username == username).first()
            db.close()
            if user_data and verify_password(password, user_data.hashed_password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success(f"Welcome, {username}!")
                st.experimental_rerun()
            else:
                st.error("Incorrect username or password.")

def register_form():
    st.subheader("Register New User")
    with st.form("register_form"):
        new_username = st.text_input("New Username")
        new_password = st.text_input("New Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submitted = st.form_submit_button("Register")

        if submitted:
            if new_password != confirm_password:
                st.error("Passwords do not match.")
            elif len(new_password) < 6:
                st.error("Password must be at least 6 characters long.")
            else:
          
                db_user = db.query(DBUser).filter(DBUser.username == new_username).first()
                if db_user:
                    st.error("Username already registered")
                    db.close()
                else:
                    hashed_password = get_password_hash(new_password)
                    new_db_user = DBUser(username=new_username, hashed_password=hashed_password)
                    db.add(new_db_user)
                    db.commit()
                    db.refresh(new_db_user)
                    db.close()
                    st.success("Registration successful! Please login.")
                    st.session_state.logged_in = False
                    st.experimental_rerun()

if not st.session_state.logged_in:
    st.sidebar.title("Authentication")
auth_option = st.sidebar.radio("", ["Login", "Register"])
  # --- Main App Content (only shown if logged in) ---

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
      


model = load_crop_yield_model()


# ---------------- FUNCTIONS ----------------
def predict_yield(input_df):
    if model is None:
        return [np.random.uniform(1000, 5000)]

    cols = ['Area', 'Item', 'Year', 'rainfall', 'pesticides', 'temperature']
    input_df = input_df[cols]

    pred = model.predict(input_df)
    return np.expm1(pred)


def get_openai_response(prompt):
    if openai_client is None:
        return "⚠️ OpenAI not configured. Add your API key."

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You are an agricultural expert helping farmers in Zambia.
                    Give simple, practical, and actionable farming advice."""
                },
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content

    except Exception as e:
        return get_openai_response(prompt)


# ---------------- UI ----------------
st.set_page_config(page_title="Crop Advisory System", layout="wide")
st.title("🌾 AI Crop Advisory & Yield Prediction System")

tabs = st.tabs([
    "📈 Yield Prediction",
    "🐛 Pest Detection",
    "🦾 AI Advisor",
    "🛒Farmer Market & Trends"
])

# =======================
# 📈 YIELD PREDICTION
# =======================
with tabs[0]:
    st.header("Crop Yield Prediction")

    with st.form("prediction_form"):
        area = st.selectbox("Area", ["Zambia", "Zimbabwe"])
        crop = st.selectbox("Crop", ["Wheat", "Maize", "Rice", "Sorghum", "Soybeans"])
        year = st.number_input("Year", 2024, 2035, 2025)
        rainfall = st.slider("Rainfall (mm)", 0, 3000, 1000)
        pesticides = st.slider("Pesticides", 0, 10000, 2000)
        temperature = st.slider("Temperature (°C)", 10, 40, 25)

        submit = st.form_submit_button("Predict")

        if submit:
            data = pd.DataFrame([{
                "Area": area,
                "Item": crop,
                "Year": year,
                "rainfall": rainfall,
                "pesticides": pesticides,
                "temperature": temperature
            }])

            prediction = predict_yield(data)
            st.success(f"🌾 Estimated Yield: {prediction[0]:,.2f} hg/ha")


# =======================
# 🐛 PEST DETECTION
# =======================
with tabs[1]:
    st.header("Pest Detection")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])

    if uploaded_file:
        st.image(uploaded_file)

        

        img = Image.open(uploaded_file).resize((224, 224))
        
        

        

        st.subheader("Results:")
        for R in "Results":
            st.write("f{R[1]} ({R[2]:.2f})")


# =======================
# 🦾 AI ADVISOR (OPENAI)
# =======================
with tabs[2]:
    st.header('AI crop Advisor')
    ai_provider = st.radio("Choose AI Provider:", ("Gemini", "OpenAI"), key="ai_provider_selector")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display messages from history
    for m in st.session_state.messages:
        with st.chat_message(m['role']):
            st.markdown(m['content'])

    if prompt := st.chat_input('How can I improve my soil?'):
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        st.chat_message('user').write(prompt)

        with st.spinner("Getting advice from the AI..."):
            response_text = "Error: AI provider not configured."
            if ai_provider == "Gemini":
                if gemini_llm:
                    response_text = asyncio.run(get_gemini_response(prompt, gemini_llm))
                else:
                    st.error("Gemini model not available. Please check API key configuration ('yield_key').")
                    response_text = "Error: Gemini model not available."
            elif ai_provider == "OpenAI":
                if openai_client:
                    response_text = asyncio.run(get_openai_response(prompt, openai_client))
                else:
                    st.error("OpenAI client not available. Please set 'OPENAI_API_KEY' in Colab secrets.")
                    response_text = "Error: OpenAI client not available."

            st.session_state.messages.append({'role': 'assistant', 'content': response_text})
    
                                            
                                                        
        
 # =======================
# Farmer Market & Trending Crops
# =======================
with tabs[3]:
    st.header("🛒 Farmer Marketplace & Trending Crops")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Trending Crops This Season')
        if os.path.exists("market trends"):
            df_trends = pd.read_csv("market trends")
            st.dataframe(df_trends, use_container_width=True)
        else: st.info('No trend data available.')
    with col2:
        st.subheader('List Your Crop for Sale')
        with st.form('market_form'):
            seller_name = st.text_input('Name','contact')
            crop_type = st.selectbox('Crop', ['Wheat', 'Maize', 'Potatoes', 'Rice, paddy', 'Sorghum', 'Soybeans'])
            quantity = st.number_input('Quantity (kg)', min_value=1)
            price = st.number_input('Asking Price (ZMW)', min_value=50)
            if st.form_submit_button('Post Listing'):
                st.success(f'Listing created for {seller_name}.')
