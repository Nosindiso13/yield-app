import streamlit as st
import requests # For API calls to FastAPI backend
import json
import pandas as pd
import numpy as np
import os
import io
from PIL import Image
import asyncio

# --- Configuration ---
API_BASE_URL = "http://127.0.0.1:8000" # Ensure this matches your FastAPI host/port
TOKEN_URL = f"{API_BASE_URL}/token"
REGISTER_USER_URL = f"{API_BASE_URL}/register_user"
PREDICT_API_URL = f"{API_BASE_URL}/predict"
CHAT_API_URL = f"{API_BASE_URL}/chat"
DETECT_PEST_API_URL = f"{API_BASE_URL}/detect_pest"

MARKET_PATH = 'market_trends.csv' # Still used locally for market data

# --- Helper for authenticated requests ---
def get_auth_headers():
    if 'access_token' in st.session_state and 'token_type' in st.session_state:
        return {"Authorization": f"{st.session_state.token_type} {st.session_state.access_token}"}
    return {}

# --- Streamlit App Setup ---
st.set_page_config(page_title='Farmer Advisor & Marketplace', layout='wide')
st.title('🌾 Crop Advisor & Marketplace')

# --- Authentication Logic ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.user_role = None # Role might be retrieved from FastAPI later if needed
    st.session_state.access_token = None
    st.session_state.token_type = None

def login():
    st.subheader("Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            try:
                response = requests.post(TOKEN_URL,
                                        data={
                                            "username": username,
                                            "password": password
                                        })
                response.raise_for_status() # Raise an exception for HTTP errors
                token_data = response.json()

                st.session_state.logged_in = True
                st.session_state.username = username # FastAPI doesn't return full user object in token, so store username
                st.session_state.user_role = "user" # Default role for now, or fetch from /users/me endpoint if available
                st.session_state.access_token = token_data['access_token']
                st.session_state.token_type = token_data['token_type']
                st.success(f"Welcome, {username}!")
                st.experimental_rerun() # Rerun to show authenticated content

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 401:
                    st.error("Incorrect username or password.")
                else:
                    st.error(f"Login error: {e.response.json().get('detail', 'Unknown error')}")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the authentication server. Please ensure FastAPI is running.")
            except Exception as e:
                st.error(f"An unexpected error occurred during login: {e}")

def register():
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
                try:
                    response = requests.post(REGISTER_USER_URL,
                                            json={
                                                "username": new_username,
                                                "hashed_password": new_password # FastAPI hashes it internally
                                            })
                    response.raise_for_status() # Raise an exception for HTTP errors
                    st.success("Registration successful! Please login.")
                    st.session_state.logged_in = False # Force login after registration
                    st.experimental_rerun()
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 400:
                        st.error(e.response.json().get('detail', 'Username already exists.'))
                    else:
                        st.error(f"Registration error: {e.response.json().get('detail', 'Unknown error')}")
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to the registration server. Please ensure FastAPI is running.")
                except Exception as e:
                    st.error(f"An unexpected error occurred during registration: {e}")

if not st.session_state.logged_in:
    st.sidebar.title("Authentication")
    auth_option = st.sidebar.radio("", ["Login", "Register"])
    if auth_option == "Login":
        login()
    else:
        register()
    st.stop() # Stop execution if not logged in

# --- Main App Content (only shown if logged in) ---
st.sidebar.write(f"Logged in as: {st.session_state.username}") # Role not strictly needed here
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.user_role = None
    st.session_state.access_token = None
    st.session_state.token_type = None
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
            input_data = {
                "Area": area,
                "Item": item,
                "Year": year,
                "rainfall": float(rain),
                "pesticides": float(pest),
                "temperature": float(temp)
            }
            try:
                response = requests.post(PREDICT_API_URL,
                                        headers=get_auth_headers(),
                                        json=[input_data]) # FastAPI expects a list
                response.raise_for_status()
                predictions = response.json()['predictions'][0]
                st.success(f'Estimated Yield: {predictions:,.2f} hg/ha')
            except requests.exceptions.HTTPError as e:
                st.error(f"Prediction error: {e.response.json().get('detail', 'Unknown error')}")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the prediction server. Please ensure FastAPI is running.")
            except Exception as e:
                st.error(f"An unexpected error occurred during prediction: {e}")

with tabs[1]:
    st.header('Pest Identification')
    file = st.file_uploader('Upload leaf or pest image', type=['jpg', 'png'])
    if file:
        st.image(file, caption="Uploaded Image", use_column_width=True)
        st.write('Detecting pest...')
        try:
            files = {'file': (file.name, file.getvalue(), file.type)}
            response = requests.post(DETECT_PEST_API_URL,
                                    headers=get_auth_headers(),
                                    files=files)
            response.raise_for_status()
            detections = response.json()['detections']
            if detections:
                st.success("Detection Results:")
                for detection in detections:
                    st.write(f"- **{detection['description']}** (Confidence: {detection['probability']:.2f})")
            else:
                st.info("No significant detections found.")
        except requests.exceptions.HTTPError as e:
            st.error(f"Pest detection error: {e.response.json().get('detail', 'Unknown error')}")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the pest detection server. Please ensure FastAPI is running.")
        except Exception as e:
            st.error(f"An unexpected error occurred during pest detection: {e}")

with tabs[2]:
    st.header('AI Chatbot')
    if 'messages' not in st.session_state: st.session_state.messages = []
    for m in st.session_state.messages: st.chat_message(m['role']).write(m['content'])
    if prompt := st.chat_input('How can I improve my soil?'):
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        st.chat_message('user').write(prompt)
        try:
            response = requests.post(CHAT_API_URL,
                                    headers=get_auth_headers(),
                                    json={"message": prompt})
            response.raise_for_status()
            ai_response = response.json()['response']
            st.session_state.messages.append({'role': 'assistant', 'content': ai_response})
            st.chat_message('assistant').write(ai_response)
        except requests.exceptions.HTTPError as e:
            st.error(f"AI Chat error: {e.response.json().get('detail', 'Unknown error')}")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the AI chat server. Please ensure FastAPI is running.")
        except Exception as e:
            st.error(f"An unexpected error occurred during AI chat: {e}")

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
