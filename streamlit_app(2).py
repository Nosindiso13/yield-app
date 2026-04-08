import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image


# ==============================
# 🔐 OPENAI CLIENT (FIXED)
# ==============================

# ==============================
# DATABASE SETUP
# ==============================
DATABASE_URL = "sqlite:///users.db"



# ==============================
# AUTH FUNCTIONS
# ==============================
def register_user(username, password):
   
    hashed_password = bcrypt.hash(password)
    user = User(username=username, password=hashed_password)
    db.add(user)
    db.commit()
    db.close()

def login_user(username, password):
    
    user = db.query(User).filter(User.username == username).first()
    db.close()
    if user and bcrypt.verify(password, user.password):
        return True
    return False

# ==============================
# LOAD MODEL (YIELD)
# ==============================
MODEL_PATH = "model_artifacts/xgboost_pipeline.joblib"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

model = load_model()

# ==============================
# AI RESPONSE FUNCTION
# ==============================
def get_ai_response(prompt):
    try:
        if openai_client is None:
            return "⚠️ OpenAI not configured."

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an agricultural expert."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content

    except Exception as e:
        # 🔁 OFFLINE FALLBACK
        return f"⚠️ AI unavailable. Basic advice: Ensure proper irrigation, pest control, and soil fertility.\n\nError: {str(e)}"

# ==============================
# UI START
# ==============================
st.set_page_config(page_title="AI Crop Advisory System", layout="wide")

# SESSION STATE
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ==============================
# LOGIN / REGISTER
# ==============================
if not st.session_state.logged_in:
    st.title("🔐 Login / Register")

    choice = st.radio("Select Option", ["Login", "Register"])

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if choice == "Register":
        if st.button("Register"):
            
            st.success("User registered! Please login.")

    if choice == "Login":
        if st.button("Login"):
            
                st.session_state.logged_in = True
                st.success("Login successful!")
                st.rerun()
         
                st.error("Invalid credentials")

# ==============================
# MAIN APP
# ==============================
else:
    st.title("🌱 AI Crop Advisory System")

    tabs = st.tabs([
        "🌾 Yield Prediction",
        "🐛 Pest Detection",
        "🤖 AI Advisor",
        "📊 Tranding crops",
        "🛒 Market Place"
    ])

    # ==============================
    # TAB 1: YIELD
    # ==============================
   # =======================
# 📈 YIELD PREDICTION
# =======================
with tabs[0]:
    st.header("Crop Yield Prediction")

    with st.form("prediction_form"):
        area = st.selectbox("Area", ["Zambia", "Zimbabwe"])
        crop = st.selectbox("Crop", ["Wheat", "Maize", "Rice", "Sorghum", "Soybeans"])
        year = st.number_input("Year", 2024, 2035, 2025)
        soil_quality = st.slider("Soil Quality", 1, 10)
        rainfall = st.slider("Rainfall (mm)", 0, 3000, 1000)
        pesticides = st.slider("Pesticides", 0, 10000, 2000)
        temperature = st.slider("Temperature (°C)", 10, 40, 25)

        submit = st.form_submit_button("Predict")

        if submit:
            data = pd.DataFrame([{
                "Area": area,
                "Item": crop,
                "soil_quality": soil_quality,
                "Year": year,
                "rainfall": rainfall,
                "pesticides": pesticides,
                "temperature": temperature
            }])

      
                


    # ==============================
    # TAB 2: PEST DETECTION
    # ==============================
    with tabs[1]:
        st.header("Pest Detection")

        uploaded_file = st.file_uploader("Upload crop image", type=["jpg", "png"])

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image")

            st.info("🔍 Analyzing image...")
            st.success("No major pest detected (demo result)")

    # ==============================
    # TAB 3: AI ADVISOR
    # ==============================
    with tabs[2]:
        st.header("AI Crop Advisor")

        user_input = st.text_area("Ask anything about crops")

        if st.button("Get Advice"):
            if user_input:
                response = get_ai_response(user_input)
                st.write(response)

    # ==============================
    # TAB 4: Trending Crops
    # ==============================
    with tabs[3]:
        st.header("Trending Crops")

        data = {
            "Crop": ["Maize", "Wheat", "Soybeans"],
            "Price (ZMW)": [250, 300, 400]
        }

        df = pd.DataFrame(data)
        st.table(df)

    # ==============================
    # TAB 5: Farmer Market 
    # ==============================
    with tabs[4]:
       st.header("🛒 Farmer Marketplace")
      
    with st.form("market_form"):
       seller_name = st.text_input('Name','contact')
       crop_type = st.selectbox('Crop', ['Wheat', 'Maize', 'Potatoes', 'Rice, paddy', 'Sorghum', 'Soybeans'])
       quantity = st.number_input('Quantity (kg)', min_value=1)
       price = st.number_input('Asking Price (ZMW)', min_value=50)
       if st.form_submit_button('Post Listing'):
       st.success(f'Listing created for {seller_name}.')

    # LOGOUT
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()
