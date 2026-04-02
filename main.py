from pydantic import BaseModel, Field
from typing import List, Dict
import pandas as pd
import numpy as np
import joblib
import os
import io
from PIL import Image
from datetime import datetime, timedelta

# FastAPI imports
from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

# Authentication imports
from passlib.context import CryptContext
from jose import JWTError, jwt

# Import for Gemini API
import google.generativeai as genai

# New Tensorflow imports for pest detection
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# --- 0. Configure Gemini & Security ---
# In a separate process, we use environment variables instead of userdata.get()
GOOGLE_API_KEY = os.getenv('YIELD_API_KEY')
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# Security settings
SECRET_KEY = os.getenv("SECRET_KEY", "a_very_secret_key_that_should_be_changed_in_production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token") # This tokenUrl points to our /login endpoint

# --- 1. Pydantic Models ---

# User models
class User(BaseModel):
    username: str
    email: str | None = None
    full_name: str | None = None
    disabled: bool | None = None
    hashed_password: str # This should ideally not be exposed directly in responses

class UserInDB(User):
    pass

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: str | None = None

class CropYieldRequest(BaseModel):
    Area: str
    Item: str
    Year: int = Field(..., gt=0)
    rainfall: float = Field(..., ge=0.0)
    pesticides: float = Field(..., ge=0.0)
    temperature: float = Field(..., ge=-50.0, le=70.0)

class ChatRequest(BaseModel):
    message: str

# --- 2. FastAPI App Instance ---
app = FastAPI(title="Crop Yield & Farmer Advisory API")

# --- 3. In-memory 'Database' & User Management Functions ---
# For demonstration, a simple dictionary to store users
fake_users_db = {}

# Helper functions for password hashing
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

# Helper functions for JWT tokens
def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    # In a real app, you would fetch the user from a database here
    # For now, we'll just check if the username exists in our fake_users_db (after registration)
    if token_data.username not in fake_users_db:
        raise credentials_exception
    return fake_users_db[token_data.username]

# --- 4. Global Models ---
model_pipeline = None
gemini_model = None
pest_model = None
MODEL_PATH = 'model_artifacts/xgboost_pipeline.joblib'

@app.on_event('startup')
async def load_models():
    global model_pipeline, gemini_model, pest_model
    if not os.path.exists(MODEL_PATH):
        # Handle missing model more gracefully, maybe raise a warning but don't stop startup
        print(f"Warning: Model file not found at {MODEL_PATH}. Prediction endpoint might not work.")
        model_pipeline = None # Set to None if not found
    else:
        try:
            model_pipeline = joblib.load(MODEL_PATH)
        except Exception as e:
            print(f"Error loading crop yield model: {e}")
            model_pipeline = None

    if GOOGLE_API_KEY:
        try:
            gemini_model = genai.GenerativeModel('gemini-pro')
        except Exception as e:
            print(f"Error loading Gemini model: {e}")
            gemini_model = None
    else:
        print("Warning: YIELD_API_KEY not set. Chat endpoint might not work.")
        gemini_model = None

    try:
        pest_model = MobileNetV2(weights='imagenet')
    except Exception as e:
        print(f"Error loading MobileNetV2 model: {e}")
        pest_model = None

    print("All models attempted to load.")

# --- 5. API Endpoints ---

@app.get('/')
async def read_root():
    return {'status': 'active', 'message': 'API is running'}

@app.post('/predict')
async def predict(request_data: List[CropYieldRequest], current_user: User = Depends(get_current_user)): # Secured endpoint
    if model_pipeline is None: raise HTTPException(status_code=500, detail="Model not loaded")
    df = pd.DataFrame([item.model_dump() for item in request_data])
    log_predictions = model_pipeline.predict(df)
    return {'predictions': np.expm1(log_predictions).tolist()}

@app.post('/chat')
async def chat(request: ChatRequest, current_user: User = Depends(get_current_user)): # Secured endpoint
    if not gemini_model:
        raise HTTPException(status_code=500, detail="Gemini API not configured")
    response = await gemini_model.generate_content_async(request.message)
    return {'response': response.text}

@app.post('/detect_pest')
async def detect_pest_endpoint(file: UploadFile = File(...), current_user: User = Depends(get_current_user)): # Secured endpoint
    if pest_model is None: raise HTTPException(status_code=500, detail="Pest detection model not loaded")
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).resize((224, 224))
    x = preprocess_input(np.expand_dims(image.img_to_array(img), axis=0))
    preds = decode_predictions(pest_model.predict(x), top=3)[0]
    return {"detections": [{"label": l, "description": d, "probability": float(p)} for (_, l, p) in preds]}

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    # This is a placeholder for actual user authentication against a database
    # In a real app, you would verify the username and hashed password from your DB
    user_data = fake_users_db.get(form_data.username)
    if not user_data or not verify_password(form_data.password, user_data["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # For demo, assign a role. In a real app, this would come from user_data
    user = UserInDB(**user_data)
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "scopes": form_data.scopes},
        expires_delta=access_token_expires,
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/register_user")
async def register_user(user: User):
    if user.username in fake_users_db:
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed_password = get_password_hash(user.hashed_password)
    fake_users_db[user.username] = user.model_dump()
    fake_users_db[user.username]["hashed_password"] = hashed_password
    return {"message": "User registered successfully"}

