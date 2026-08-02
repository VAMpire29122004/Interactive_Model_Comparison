# -*- coding: utf-8 -*-
"""mnist_write_app.py"""

import streamlit as st
import numpy as np
import cv2
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, Model 
from streamlit_drawable_canvas import st_canvas
import datetime
import gspread
from google.oauth2.service_account import Credentials
import time

# --- 1. Page Config & Title ---
st.set_page_config(page_title="MNIST Model Comparison", layout="wide")
st.title("Draw a Digit: Comparing Models")
st.write("Draw a number from 0-9 below and see how different models interpret it!")


# --- 2. GOOGLE SHEETS AUTHENTICATION & LOGGING ---
@st.cache_resource
def get_gsheet_client():
    """Authenticates with Google Sheets API using Streamlit Secrets."""
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    credentials = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=scopes,
    )
    return gspread.authorize(credentials)


def log_to_google_sheet(sheet_name, model_used, predicted_label, actual_label, pixel_array):
    """Appends a new row to Google Sheets with prediction metadata and pixel data."""
    try:
        client = get_gsheet_client()
        sheet = client.open(sheet_name).sheet1

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Convert the 784 numpy array/list to string so Google Sheets can store it
        pixel_data_str = (
            str(pixel_array.tolist())
            if hasattr(pixel_array, "tolist")
            else str(pixel_array)
        )

        # Append row: [Timestamp, Selected Model, Model Prediction, Correct Label, Image Pixels]
        sheet.append_row(
            [timestamp, model_used, int(predicted_label), int(actual_label), pixel_data_str],
            value_input_option="USER_ENTERED"
        )
        return True
        
    except gspread.exceptions.SpreadsheetNotFound:
        st.error(f"Could not find Sheet '{sheet_name}'. Did you share it with the bot email?")
        return False
    except Exception as e:
        # Bypass the false alarm if the server successfully responded with a 200 OK code
        if "200" in str(e):
            return True
        st.error(f"Failed to log data: {e}")
        return False


# --- 3. Load Models (Cached for speed) ---
@st.cache_resource
def load_all_models():
    # Load the Scikit-Learn models
    rf = joblib.load('rf_model_mnistdataset.joblib')
    l1 = joblib.load('l1_model_mnistdataset.joblib')

    # Load the Keras Sequential Models
    nn1 = load_model('L1R128_L2sig128_L3smax10.keras')
    nn2 = load_model('L1R264_L2sig264_L3smax10.keras')

    return rf, l1, nn1, nn2

rf_model, l1_model, nn_model_1, nn_model_2 = load_all_models()


# --- 4. Initialize Session State ---
# This ensures the app remembers the prediction when the user clicks the feedback buttons
if "prediction_made" not in st.session_state:
    st.session_state.prediction_made = False
if "current_input_data" not in st.session_state:
    st.session_state.current_input_data = None
if "current_prediction" not in st.session_state:
    st.session_state.current_prediction = None
if "current_model_choice" not in st.session_state:
    st.session_state.current_model_choice = None
if "canvas_key" not in st.session_state:         
    st.session_state.canvas_key = 0              


# --- 5. UI Layout ---
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### Drawing Pad")
    # Create an interactive canvas using the dynamic key
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=15, # Thick brush for better 28x28 scaling
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key=f"canvas_{st.session_state.canvas_key}", 
    )

    model_choice = st.selectbox(
        "Choose your model:",
        ("Random Forest", "L1- Lasso Model", "Sequential NN (Model 1)", "Sequential NN (Model 2)")
    )

    predict_btn = st.button("Predict Digit", type="primary")
    
    # --- FEEDBACK WIDGET ---
    if st.session_state.prediction_made:
        st.divider()
        st.subheader("🤖 Help Us Improve (Data Flywheel)")
        st.write("Was this prediction accurate?")

        col_fb1, col_fb2 = st.columns([1, 1])

        with col_fb1:
            is_correct = st.radio("Is prediction correct?", ("Yes", "No"), index=0, key="feedback_radio")

        if is_correct == "No":
            with col_fb2:
                actual_digit = st.number_input("What was the actual digit?", min_value=0, max_value=9, step=1)
                if st.button("Submit Correct Label"):
                    success = log_to_google_sheet(
                        sheet_name="MNIST_Feedback_Data",  
                        model_used=st.session_state.current_model_choice,
                        predicted_label=st.session_state.current_prediction,
                        actual_label=actual_digit,
                        pixel_array=st.session_state.current_input_data,
                    )
                    if success:
                        st.success("Logged! Thank you for feeding the data flywheel. 🚀")
                        
                        # --- RESET LOGIC ---
                        time.sleep(1.5) 
                        st.session_state.prediction_made = False 
                        st.session_state.canvas_key += 1 
                        st.rerun() 

        elif is_correct == "Yes":
            with col_fb2:
                if st.button("Log Correct Sample"):
                    success = log_to_google_sheet(
                        sheet_name="MNIST_Feedback_Data",
                        model_used=st.session_state.current_model_choice,
                        predicted_label=st.session_state.current_prediction,
                        actual_label=st.session_state.current_prediction,
                        pixel_array=st.session_state.current_input_data,
                    )
                    if success:
                        st.success("Logged! Thanks for confirming. 🎉")
                        
                        # --- RESET LOGIC ---
                        time.sleep(1.5) 
                        st.session_state.prediction_made = False 
                        st.session_state.canvas_key += 1 
                        st.rerun() 


with col2:
    st.markdown("### Prediction Results")

    # --- 6. Process Drawing & Predict ---
    if predict_btn and canvas_result.image_data is not None:
        # The canvas outputs a 280x280 RGBA image. We need it to be 28x28 Grayscale.
        img_array = canvas_result.image_data
        gray_image = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)
        resized_image = cv2.resize(gray_image, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Flatten for models (1, 784) and normalize
        input_data = resized_image.reshape(1, 784) / 255.0
        
        # Save to session state
        st.session_state.current_input_data = input_data
        st.session_state.current_model_choice = model_choice
        st.session_state.prediction_made = True

        # --- Model Prediction Logic ---
        if model_choice == "Random Forest":
            prediction = rf_model.predict(input_data)[0]
            st.session_state.current_prediction = prediction
            
        elif model_choice == "L1- Lasso Model":
            prediction = l1_model.predict(input_data)[0]
            st.session_state.current_prediction = prediction
            
        elif model_choice == "Sequential NN (Model 1)":
            probabilities = nn_model_1.predict(input_data.reshape(1,28,28))[0]
            prediction = np.argmax(probabilities)
            st.session_state.current_prediction = prediction
            st.session_state.probabilities = probabilities
            
        elif model_choice == "Sequential NN (Model 2)":
            probabilities = nn_model_2.predict(input_data.reshape(1,28,28))[0]
            prediction = np.argmax(probabilities)
            st.session_state.current_prediction = prediction
            st.session_state.probabilities = probabilities

    # --- 7. Display Results & Visualizations (Driven by Session State) ---
    if st.session_state.prediction_made:
        model_choice = st.session_state.current_model_choice
        prediction = st.session_state.current_prediction
        input_data = st.session_state.current_input_data

        if model_choice == "Random Forest":
            st.success(f"### Random Forest predicts: **{prediction}**")
            st.markdown("### What the Forest is looking at:")
            importances = rf_model.feature_importances_.reshape(28, 28)
            fig, ax = plt.subplots()
            cax = ax.imshow(importances, cmap='hot', interpolation='nearest')
            fig.colorbar(cax)
            st.pyplot(fig)

        elif model_choice == "L1- Lasso Model":
            st.success(f"### L1- Lasso Model predicts: **{prediction}**")
            st.markdown(f"### Visualizing the process of drawing to {prediction}?")
            coefs = l1_model.coef_[int(prediction)].reshape(28, 28)
            fig, ax = plt.subplots()
            cax = ax.imshow(coefs, cmap='coolwarm', interpolation='nearest') 
            fig.colorbar(cax)
            st.pyplot(fig)

        elif model_choice == "Sequential NN (Model 1)":
            probabilities = st.session_state.probabilities
            st.success(f"### Sequential NN (Model 1) predicts: **{prediction}**")
            st.write(f"Confidence: {probabilities[prediction]:.2%}")
            st.progress(float(probabilities[prediction]))
            
            st.markdown("### Inside the Network: Neuron Activations")
            x = input_data.reshape(1, 28, 28)
            for i, layer in enumerate(nn_model_1.layers):
                x = layer(x) 
                activation_data = x.numpy() 
                st.caption(f"Layer {i+1}: {layer.name} ({activation_data.shape[-1]} neurons)")
                if len(activation_data.shape) == 2:
                    st.bar_chart(activation_data[0])

        elif model_choice == "Sequential NN (Model 2)":
            probabilities = st.session_state.probabilities
            st.success(f"### Sequential NN (Model 2) predicts: **{prediction}**")
            st.write(f"Confidence: {probabilities[prediction]:.2%}")
            st.progress(float(probabilities[prediction]))
            
            st.markdown("### Inside the Network: Neuron Activations")
            x = input_data.reshape(1, 28, 28)
            for i, layer in enumerate(nn_model_2.layers):
                x = layer(x)
                activation_data = x.numpy()
                st.caption(f"Layer {i+1}: {layer.name} ({activation_data.shape[-1]} neurons)")
                if len(activation_data.shape) == 2:
                    st.bar_chart(activation_data[0])
