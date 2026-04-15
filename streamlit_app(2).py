import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
import json
from PIL import Image
# ==============================
# 🔐 SECURE API KEY SETUP
# ==============================
# API key is loaded from Streamlit secrets (st.secrets) OR environment variable.
# NEVER hardcode your key here. See README_SECRETS.md for setup instructions.

def get_openrouter_key():
    # Tries both secret names — use whichever you set in Streamlit Cloud
    for name in ["Cropkey", "OPENROUTER_API_KEY"]:
        try:
            key = st.secrets[name]
            if key:
                return key
        except Exception:
            pass
    # Fallback: environment variable (local dev)
    return os.environ.get("Cropkey") or os.environ.get("OPENROUTER_API_KEY")

OPENROUTER_API_KEY = get_openrouter_key()
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "openrouter/free"  # Auto-selects any available free model — never breaks
# ==============================
# DATABASE SETUP (SQLite + SQLAlchemy)
# ==============================


DATABASE_URL = "sqlite:///users.db"
               

    
   



# ==============================
# AUTH FUNCTIONS (FIXED)
# ==============================
def register_user(username, password):

    try:
        existing = db.query(User).filter(User.username == username).first()
        if existing:
            return False, "Username already exists."
        hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        user = User(username=username, password=hashed_password)
        db.add(user)
        db.commit()
        return True, "Registered successfully!"
    except Exception as e:
        return False, str(e)
    finally:
        db.close()

def login_user(username, password):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == username).first()
        if user and bcrypt.checkpw(password.encode(), user.password.encode()):
            return True
        return False
    except Exception:
        return False
    finally:
        db.close()

# ==============================
# LOAD YIELD MODEL (SAFE)
# ==============================
MODEL_PATH = "model_artifacts/xgboost_pipeline.joblib"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

model = load_model()

def predict_yield(data: pd.DataFrame):
    if model is not None:
        return model.predict(data)
    # Demo fallback: simple heuristic
    base = {"Maize": 3500, "Wheat": 2800, "Rice": 4200, "Sorghum": 2100, "Soybeans": 1900}
    crop = data["Item"].iloc[0]
    rainfall_factor = data["rainfall"].iloc[0] / 1000
    temp_factor = 1 - abs(data["temperature"].iloc[0] - 25) / 50
    return [base.get(crop, 3000) * rainfall_factor * temp_factor]

# ==============================
# 🤖 AI AGENT via OpenRouter (FIXED & ENHANCED)
# ==============================
def call_openrouter(system_prompt: str, user_prompt: str) -> str:
    if not OPENROUTER_API_KEY:
        return (
            "⚠️ OpenRouter API key not configured.\n\n"
            
        )
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://ai-crop-advisory.streamlit.app",
            "X-Title": "AI Crop Advisory System"
        }
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 800
        }
        resp = requests.post(OPENROUTER_BASE_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.Timeout:
        return "⚠️ Request timed out. Please try again."
    except requests.exceptions.HTTPError as e:
        return f"⚠️ API error: {e.response.status_code} — {e.response.text}"
    except Exception as e:
        return f"⚠️ Unexpected error: {str(e)}"

def agent_yield_advice(crop, area, rainfall, temperature, pesticides, predicted_yield):
    system = (
        "You are an expert agricultural advisor for sub-Saharan Africa. "
        "Give practical, actionable advice. Be concise and clear."
    )
    prompt = (
        f"Crop: {crop}, Region: {area}, Rainfall: {rainfall}mm, "
        f"Temp: {temperature}°C, Pesticides used: {pesticides}g/ha, "
        f"Predicted yield: {predicted_yield:,.0f} hg/ha. "
        "Give 3 specific tips to improve this yield."
    )
    return call_openrouter(system, prompt)

def agent_pest_detection(crop, symptoms):
    system = (
        "You are a plant pathologist specializing in African agriculture. "
        "Identify likely pests/diseases and give treatment advice."
    )
    prompt = f"Crop: {crop}. Observed symptoms: {symptoms}. What pests or diseases are likely? How should the farmer treat them?"
    return call_openrouter(system, prompt)

def agent_trending_crops(region, season):
    system = (
        "You are an agricultural market analyst for southern Africa. "
        "Give practical market trend advice for smallholder farmers."
    )
    prompt = f"Region: {region}, Season: {season}. Which 3 crops are most profitable right now and why? Include price trends."
    return call_openrouter(system, prompt)

def agent_market_advisor(crop, quantity, region):
    system = (
        "You are a farmers market expert helping smallholder farmers in southern Africa "
        "find the best channels to sell their produce."
    )
    prompt = (
        f"Farmer has {quantity} kg of {crop} in {region}. "
        "Suggest the best markets, buyers, or cooperatives to contact. "
        "Include tips on pricing and negotiation."
    )
    return call_openrouter(system, prompt)

def agent_general_advisor(question):
    system = (
        "You are a knowledgeable agricultural advisor for African smallholder farmers. "
        "Answer questions about crops, soil, water, pests, and farming practices. "
        "Be practical, friendly, and specific to the southern Africa context."
    )
    return call_openrouter(system, question)

def get_my_listings(crop_filter=None, location_filter=None):
    # your database query or API call here
    # return a list of listings
    return[]

def add_listing(username, farmer_name, crop, quantity_kg, price_per_kg, location,contact, description):
    #your code to save to database/ CSV /API
    # Must return twoo values: success boolean + message string
    try:
        #example: save to database
        return True,"Listing posted successfully"
    except Exception as e:
        return False, f"Error: {e}"
        
    

# ==============================
# UI SETUP
# ==============================
st.set_page_config(page_title="AI Crop Advisory System", layout="wide", page_icon="🌱")

st.markdown("""
<style>
    .stTabs [data-baseweb="tab"] { font-size: 15px; font-weight: 600; }
    .metric-card {
        background: linear-gradient(135deg, #1a4a2e, #2d7a4f);
        border-radius: 12px; padding: 20px; color: white; text-align: center;
    }
    .ai-response {
        background: #2c1a0e; border-left: 4px solid #a0522d;
        padding: 16px; border-radius: 8px; margin-top: 12px;
        color: #f5e6d3;
    }
</style>
""", unsafe_allow_html=True)
# SESSION STATE
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

# ==============================
# LOGIN / REGISTER (FIXED)
# ==============================
if not st.session_state.logged_in:
    st.title("🌱 AI Crop Advisory System")
    st.subheader("🔐 Login ")

    choice = st.radio("Select Option", ["Login"])
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

   
 
    if choice == "Login":
        if st.button("Login"):
            if username and password:
               
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success("Login successful!")
                    st.rerun()
               
            else:
                st.warning("Please fill in all fields.")

# ==============================
# MAIN APP (FIXED INDENTATION + STRUCTURE)
# ==============================
else:
    col1, col2 = st.columns([6, 1])
    with col1:
        st.title(f"🌱 AI Crop Advisory — Welcome, {st.session_state.username}!")
    with col2:
        if st.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.rerun()

    tabs = st.tabs([
        "🌾 Yield Prediction",
        "🐛 Pest Detection",
        "🤖 AI Advisor",
        "📊 Trending Crops",
        "🛒 Farmers Market"
    ])

    # ==========================
    # TAB 1: YIELD PREDICTION (FIXED)
    # ==========================
    with tabs[0]:
        st.header("🌾 Crop Yield Prediction")

        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            with col1:
                area = st.selectbox("Area", ["Zambia", "Zimbabwe", "Malawi", "Tanzania", "Mozambique"])
                crop = st.selectbox("Crop", ["Maize", "Wheat", "Rice", "Sorghum", "Soybeans"])
                year = st.number_input("Year", 2024, 2035, 2025)
            with col2:
                rainfall = st.slider("Rainfall (mm)", 0, 3000, 1000)
                pesticides = st.slider("Pesticides (g/ha)", 0, 10000, 2000)
                temperature = st.slider("Temperature (°C)", 10, 40, 25)

            submit = st.form_submit_button("🔮 Predict Yield")

        if submit:
            input_data = pd.DataFrame([{
                "Area": area,
                "Item": crop,
                "Year": year,
                "rainfall": rainfall,
                "pesticides": pesticides,
                "temperature": temperature
            }])

            prediction = predict_yield(input_data)
            pred_val = prediction[0]

            col1, col2, col3 = st.columns(3)
            col1.metric("🌾 Estimated Yield", f"{pred_val:,.0f} hg/ha")
            col2.metric("📍 Region", area)
            col3.metric("🌡️ Temperature", f"{temperature}°C")

            st.success(f"Estimated Yield for {crop} in {area}: **{pred_val:,.2f} hg/ha**")

            with st.spinner("🤖 Getting AI improvement tips..."):
                advice = agent_yield_advice(crop, area, rainfall, temperature, pesticides, pred_val)
            st.markdown("### 💡 AI Yield Improvement Tips")
            st.markdown(f'<div class="ai-response">{advice}</div>', unsafe_allow_html=True)

    # ==========================
    # TAB 2: PEST DETECTION (ENHANCED)
    # ==========================
    with tabs[1]:
        st.header("🐛 Pest Detection")

        col1, col2 = st.columns([1, 1])
        with col1:
            uploaded_file = st.file_uploader("Upload crop image (optional)", type=["jpg", "jpeg", "png"])
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Crop Image", use_container_width=True)

        with col2:
            crop_type = st.selectbox("Crop Type", ["Maize", "Wheat", "Rice", "Sorghum", "Soybeans", "Other"])
            symptoms = st.text_area(
                "Describe what you observe",
                placeholder="e.g. yellowing leaves, white spots on stems, holes in leaves, wilting..."
            )
            if st.button("🔍 Detect Pests & Diseases"):
                if symptoms:
                    with st.spinner("🤖 Analyzing symptoms..."):
                        result = agent_pest_detection(crop_type, symptoms)
                    st.markdown("### 🦠 AI Pest & Disease Analysis")
                    st.markdown(f'<div class="ai-response">{result}</div>', unsafe_allow_html=True)
                else:
                    st.warning("Please describe the symptoms you observe.")

    # ==========================
    # TAB 3: AI ADVISOR (ENHANCED)
    # ==========================
    with tabs[2]:
        st.header("🤖 AI Crop Advisor")
        st.markdown("Ask anything about farming — soil, irrigation, planting, harvesting, and more.")

        user_input = st.text_area("Your question", placeholder="e.g. When is the best time to plant maize in Zambia?")

        if st.button("💬 Get Advice"):
            if user_input:
                with st.spinner("🤖 Thinking..."):
                    response = agent_general_advisor(user_input)
                st.markdown("### 🌿 AI Response")
                st.markdown(f'<div class="ai-response">{response}</div>', unsafe_allow_html=True)
            else:
                st.warning("Please enter a question.")

    # ==========================
    # TAB 4: TRENDING CROPS (ENHANCED)
    # ==========================
    with tabs[3]:
        st.header("📊 Trending Crops & Market Prices")

        col1, col2 = st.columns(2)
        with col1:
            trend_region = st.selectbox("Your Region", ["Zambia", "Zimbabwe", "Malawi", "Tanzania", "Mozambique"])
        with col2:
            season = st.selectbox("Season", ["Rainy Season (Oct–Apr)", "Dry Season (May–Sep)", "Post-Harvest"])

        # Static price table
        price_data = {
            "Crop": ["Maize", "Wheat", "Soybeans", "Rice", "Sorghum", "Groundnuts", "Sunflower"],
            "Price (ZMW/50kg)": [250, 300, 400, 350, 200, 450, 280],
            "Demand": ["🔥 High", "📈 Rising", "🔥 High", "📈 Rising", "➡️ Stable", "🔥 High", "📈 Rising"],
            "Best Market": ["ZAMACE", "FRA", "Export", "Local", "Local", "Export", "Crushing Plants"]
        }
        df = pd.DataFrame(price_data)
        st.dataframe(df, use_container_width=True)

        if st.button("🤖 Get AI Market Trend Analysis"):
            with st.spinner("🤖 Analyzing market trends..."):
                trend_advice = agent_trending_crops(trend_region, season)
            st.markdown("### 📈 AI Market Trend Insights")
            st.markdown(f'<div class="ai-response">{trend_advice}</div>', unsafe_allow_html=True)

    # ==========================
    # TAB 5: FARMERS MARKET (NEW)
    # ==========================
    with tabs[4]:
        st.header("Farmers Market — Buy & Sell Crops")
 
        market_tabs = st.tabs(["Browse Listings", "Post My Crop", "AI Market Advice", "My Listings"])
 
        CROPS = ["All", "Maize", "Wheat", "Soybeans", "Rice", "Sorghum", "Groundnuts", "Sunflower", "Other"]
        LOCATIONS = ["All", "Zambia", "Zimbabwe", "Malawi", "Tanzania", "Mozambique"]
 
        # BROWSE LISTINGS
        with market_tabs[0]:
            st.subheader("Available Crop Listings")
            st.caption("Browse crops posted by farmers ready to sell.")
            col1, col2 = st.columns(2)
            with col1:
                filter_crop = st.selectbox("Filter by Crop", CROPS, key="browse_crop")
            with col2:
                filter_loc = st.selectbox("Filter by Location", LOCATIONS, key="browse_loc")
 
            listings = get_my_listings(
                crop_filter=filter_crop if filter_crop != "All" else None, 
                location_filter=filter_loc if filter_loc != "All" else None 
         )
 
            if not listings:
                st.info("No listings yet. Be the first to post your crop!")
            else:
                st.success(f"Found **{len(listings)}** listing(s)")
                for l in listings:
                    st.markdown(f"""
<div style="background:#3D2A20;border-left:4px solid #A0522D;border-radius:8px;padding:14px;margin-bottom:10px;">
  <b style="color:#E8A838;font-size:16px;">{l.crop}</b>
  <span style="color:#F5E6D3;font-size:13px;float:right;">Listed: {l.listed_on}</span><br>
  <span style="color:#F5E6D3;">Farmer: <b>{l.farmer_name}</b> &nbsp;|&nbsp; Location: {l.location}</span><br>
  <span style="color:#F5E6D3;">Qty: <b>{l.quantity_kg:,.0f} kg</b> &nbsp;|&nbsp; Price: <b>ZMW {l.price_per_kg:.2f}/kg</b></span><br>
  <span style="color:#ccc;font-size:12px;">{l.description}</span><br>
  <span style="color:#A0522D;font-size:12px;">Contact: <b>{l.contact}</b></span>
</div>""", unsafe_allow_html=True)
 
        # POST MY CROP
        with market_tabs[1]:
            st.subheader("Register Your Crop for Sale")
            st.caption("Fill in the details below to list your crop on the marketplace.")
 
            with st.form("crop_listing_form"):
                col1, col2 = st.columns(2)
                with col1:
                    farmer_name = st.text_input("Your Full Name", placeholder="e.g. John Banda")
                    crop = st.selectbox("Crop", ["Maize", "Wheat", "Soybeans", "Rice", "Sorghum", "Groundnuts", "Sunflower", "Other"])
                    quantity_kg = st.number_input("Quantity Available (kg)", min_value=1.0, value=100.0)
                with col2:
                    price_per_kg = st.number_input("Your Price (ZMW per kg)", min_value=0.1, value=5.0)
                    location = st.selectbox("Your Location", ["Zambia", "Zimbabwe", "Malawi", "Tanzania", "Mozambique"])
                    contact = st.text_input("Contact (phone or email)", placeholder="e.g. +260 97 1234567")
                description = st.text_area("Description (optional)", placeholder="e.g. Sun-dried maize, ready for collection.")
 
                submitted = st.form_submit_button("Post My Listing")
                if submitted:
                    if not farmer_name or not contact:
                        st.warning("Please fill in your name and contact details.")
                    else:
                        success, msg = add_listing(
                            username=st.session_state.username,
                            farmer_name=farmer_name, crop=crop,
                            quantity_kg=quantity_kg, price_per_kg=price_per_kg,
                            location=location, contact=contact, description=description
                        )
                        if success:
                            st.success(f"{msg} Your crop is now visible to buyers.")
                            st.balloons()
                        else:
                            st.error(msg)
 
        # AI MARKET ADVICE
        with market_tabs[2]:
            st.subheader("AI Market Advice")
            col1, col2, col3 = st.columns(3)
            with col1:
                market_crop = st.selectbox("Crop", ["Maize", "Wheat", "Soybeans", "Rice", "Sorghum", "Groundnuts", "Sunflower"], key="ai_crop")
            with col2:
                quantity = st.number_input("Quantity (kg)", min_value=1, value=500, key="ai_qty")
            with col3:
                market_region = st.selectbox("Location", ["Zambia", "Zimbabwe", "Malawi", "Tanzania", "Mozambique"], key="ai_region")
 
            if st.button("Get AI Selling Advice"):
                with st.spinner("Finding best markets for you..."):
                    market_advice = agent_market_advisor(market_crop, quantity, market_region)
                st.markdown("### AI Recommendations")
                st.markdown(f'<div class="ai-response">{market_advice}</div>', unsafe_allow_html=True)
 
            st.divider()
            st.markdown("### Known Markets & Buyers")
            markets_df = pd.DataFrame({
                "Market / Buyer": ["ZAMACE", "FRA (Food Reserve Agency)", "Export Traders (ZIM)", "Local Millers", "Cooperatives"],
                "Best For": ["Maize, Soybeans", "Maize, Wheat", "Soybeans, Wheat", "Maize, Wheat", "All crops"],
                "Contact": ["www.zamace.co.zm", "FRA District Office", "AgriTraders ZW", "Local listings", "District Cooperative"]
            })
            st.dataframe(markets_df, use_container_width=True)
 
        # MY LISTINGS
        with market_tabs[3]:
            st.subheader("My Posted Listings")
            st.caption("Manage the crops you have listed for sale.")
            my_listings = get_my_listings(st.session_state.username)
 
            if not my_listings:
                st.info("You have not posted any listings yet. Go to Post My Crop to get started!")
            else:
                st.success(f"You have **{len(my_listings)}** active listing(s)")
                for l in my_listings:
                    col1, col2 = st.columns([5, 1])
                    with col1:
                        st.markdown(f"""
<div style="background:#3D2A20;border-left:4px solid #4A7C59;border-radius:8px;padding:12px;margin-bottom:8px;">
  <b style="color:#E8A838;">{l.crop}</b> — {l.quantity_kg:,.0f} kg @ ZMW {l.price_per_kg:.2f}/kg<br>
  <span style="color:#F5E6D3;font-size:12px;">Location: {l.location} | Listed: {l.listed_on} | Contact: {l.contact}</span>
</div>""", unsafe_allow_html=True)
                    with col2:
                        if st.button("Delete", key=f"del_{l.id}"):
                            if delete_listing(l.id, st.session_state.username):
                                st.success("Deleted!")
                                st.rerun()
 
                
