import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image


# ==============================
# 🔐 OPENAI CLIENT (FIXED)
# ==============================


openai_client = OpenAI(
    api_key=st.secrets.get("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", None)
)

# ==============================
# DATABASE SETUP
# ==============================
DATABASE_URL = "sqlite:///users.db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class User(Base):
    _tablename_ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True)
    password = Column(String)

Base.metadata.create_all(bind=engine)

# ==============================
# AUTH FUNCTIONS
# ==============================
def register_user(username, password):
    db = SessionLocal()
    hashed_password = bcrypt.hash(password)
    user = User(username=username, password=hashed_password)
    db.add(user)
    db.commit()
    db.close()

def login_user(username, password):
    db = SessionLocal()
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
            register_user(username, password)
            st.success("User registered! Please login.")

    if choice == "Login":
        if st.button("Login"):
            if login_user(username, password):
                st.session_state.logged_in = True
                st.success("Login successful!")
                st.rerun()
            else:
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
        "📊 Market Info"
    ])

    # ==============================
    # TAB 1: YIELD
    # ==============================
    with tabs[0]:
        st.header("Crop Yield Prediction")

        rainfall = st.number_input("Rainfall (mm)", 0.0)
        temperature = st.number_input("Temperature (°C)", 0.0)
        soil_quality = st.slider("Soil Quality", 1, 10)

        if st.button("Predict Yield"):
            if model:
                input_data = np.array([[rainfall, temperature, soil_quality]])
                prediction = model.predict(input_data)
                st.success(f"Predicted Yield: {prediction[0]:.2f}")
            else:
                st.warning("⚠️ Model not found. Using dummy prediction.")
                st.info(f"Estimated Yield: {rainfall * 0.3 + temperature * 0.5}")

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
    # TAB 4: MARKET
    # ==============================
    with tabs[3]:
        st.header("Market Information")

        data = {
            "Crop": ["Maize", "Wheat", "Soybeans"],
            "Price (ZMW)": [250, 300, 400]
        }

        df = pd.DataFrame(data)
        st.table(df)

    # LOGOUT
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()
