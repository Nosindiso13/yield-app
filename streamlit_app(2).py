import streamlit as st
import pandas as pd
import numpy as np
import os
import io
from PIL import Image


# ---------------- CONFIG ----------------
MODEL_PATH = 'model_artifacts/xgboost_pipeline.joblib'

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_crop_yield_model():
    if not os.path.exists(MODEL_PATH):
        st.warning("Model not found. Using demo prediction.")
        return None
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_pest_model():
    from tensorflow.keras.applications import MobileNetV2
    return MobileNetV2(weights="imagenet")

@st.cache_resource
def load_openai_client():
    try:
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    except Exception as e:
        st.error(f"OpenAI error: {e}")
        return None

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
        return f"Error: {e}"


# ---------------- UI ----------------
st.set_page_config(page_title="Crop Advisory System", layout="wide")
st.title("🌾 AI Crop Advisory & Yield Prediction System")

tabs = st.tabs([
    "📈 Yield Prediction",
    "🐛 Pest Detection",
    "🦾 AI Advisor"
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

        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
        from tensorflow.keras.preprocessing import image

        img = Image.open(uploaded_file).resize((224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = pest_model.predict(x)
        results = decode_predictions(preds, top=3)[0]

        st.subheader("Results:")
        for r in results:
            st.write(f"{r[1]} ({r[2]:.2f})")


# =======================
# 🦾 AI ADVISOR (OPENAI)
# =======================
with tabs[2]:
    st.header("AI Crop Advisor (Powered by OpenAI)")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # User input
    if prompt := st.chat_input("Ask about crops, soil, pests..."):
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        st.chat_message("user").write(prompt)

        with st.spinner("Thinking..."):
            response = get_openai_response(prompt)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })
        st.chat_message("assistant").write(response)
