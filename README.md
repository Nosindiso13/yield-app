# 🌾 Crop Advisor & Marketplace Streamlit App

This is a standalone Streamlit application that provides crop yield prediction, pest detection, AI-powered farmer advisory, and a simple marketplace for agricultural products. It's designed to assist farmers with data-driven insights and connect them within a community.

## Features

-   **Crop Yield Prediction:** Predicts crop yield based on various environmental and agricultural factors using a trained XGBoost model.
-   **Pest Detection:** Identifies potential pests from uploaded images using a MobileNetV2 deep learning model.
-   **AI Advisor:** An AI chatbot powered by Google Gemini Pro that offers agricultural advice and answers farmer queries.
-   **Farmer Marketplace & Trends:** Displays trending crop prices and allows farmers to list their crops for sale.

## Architecture

This application is designed as a standalone Streamlit app, meaning it handles all its functionalities directly without relying on a separate backend service like FastAPI. This includes:

-   **Direct Model Loading:** All machine learning models (XGBoost, MobileNetV2) and the Gemini AI model are loaded directly within the Streamlit application using `@st.cache_resource` for efficiency.
-   **Integrated User Authentication:** User registration and login are handled directly within the Streamlit app using `passlib` for password hashing and `SQLAlchemy` with an SQLite database (`sql_app_new.db`).
-   **Environment Variables for API Keys:** Sensitive information like the Google Gemini API key is managed via environment variables.

## Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download Model Artifacts:**
    Ensure you have the `model_artifacts/xgboost_pipeline.joblib` file in your repository. This file contains the trained crop yield prediction model. If you trained the model in a Colab notebook, download it from there and place it in the `model_artifacts` directory.

5.  **Prepare Market Trends Data:**
    Ensure you have the `market_trends.csv` file in your repository. This file is used for the Marketplace section.

6.  **Google Gemini API Key Configuration:**
    -   Obtain a Google Gemini API key from [Google AI Studio](https://makersuite.google.com/k/api). 
    -   Set this key as an environment variable named `YIELD_API_KEY` before running the Streamlit app. 
    
    **On Linux/macOS:**
    ```bash
    export YIELD_API_KEY='YOUR_GOOGLE_GEMINI_API_KEY'
    ```
    **On Windows (Command Prompt):**
    ```bash
    set YIELD_API_KEY=YOUR_GOOGLE_GEMINI_API_KEY
    ```
    **On Streamlit Cloud / Deployment Platforms:** Configure `YIELD_API_KEY` as a secret or environment variable through their respective dashboards.

## Running the Application

Once the setup is complete and your `YIELD_API_KEY` environment variable is set, run the Streamlit application:

```bash
streamlit run streamlit_app.py
```

This will open the application in your web browser.

## Database Notes

-   The application uses an SQLite database named `sql_app_new.db` for user authentication. This file will be created automatically when the application runs for the first time.
-   **Important for Cloud Deployments (e.g., Streamlit Cloud):** SQLite files (`.db`) are local to the running instance and are ephemeral. This means that if your application restarts or redeploys on platforms like Streamlit Cloud, all user data (including the default `testuser`) stored in `sql_app_new.db` will be lost. For persistent user data in a production environment, you would typically integrate with a cloud-based database service (e.g., PostgreSQL, MySQL).

## Default Test User

For initial testing, a default user is created on startup if the database is empty:

-   **Username:** `testuser`
-   **Password:** `testpassword123`

You can use these credentials to log in and test the application's features.

## Contact

For any questions or issues, please open an issue in this repository.
