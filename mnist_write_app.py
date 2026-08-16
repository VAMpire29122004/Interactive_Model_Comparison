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

        pixel_data_str = (
            str(pixel_array.tolist())
            if hasattr(pixel_array, "tolist")
            else str(pixel_array)
        )

        sheet.append_row(
            [timestamp, model_used, int(predicted_label), int(actual_label), pixel_data_str],
            value_input_option="USER_ENTERED"
        )
        return True
        
    except gspread.exceptions.SpreadsheetNotFound:
        st.error(f"Could not find Sheet '{sheet_name}'. Did you share it with the bot email?")
        return False
    except Exception as e:
        if "200" in str(e):
            return True
        st.error(f"Failed to log data: {e}")
        return False


# --- 3. Load Models (Cached for speed) ---
@st.cache_resource
def load_all_models():
    rf = joblib.load('rf_model_mnistdataset.joblib')
    l1 = joblib.load('l1_model_mnistdataset.joblib')
    nn1 = load_model('L1R128_L2sig128_L3smax10.keras')
    nn2 = load_model('L1R264_L2sig264_L3smax10.keras')
    nn3 = load_model('L1R256_L2tanh128_L4R256_L4smax10.keras')
    nn4 = load_model('L1R10_L2sig18_L3R24_L4R6_L3smax10.keras')
    nn5 = load_model('L1R2_L2R4_L3sig6_L4R8_L3smax10.keras')
    cnn_model_1 = load_model('model_1_cnn.keras')
    cnn_model_2 = load_model('model_2_cnn.keras')
    return rf, l1, nn1, nn2, nn3, nn4, nn5, cnn_model_1, cnn_model_2

rf_model, l1_model, nn_model_1, nn_model_2, nn_model_3, nn_model_4, nn_model_5, cnn_model_1, cnn_model_2 = load_all_models()


# --- 4. Initialize Session State ---
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


# --- 5. UI Layout: Left Column (Drawing) ---
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### Drawing Pad")
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=15, 
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key=f"canvas_{st.session_state.canvas_key}", 
    )

    model_choice = st.selectbox(
        "Choose your model:",
        ("Random Forest", "L1- Lasso Model", "Sequential Neural Network 1 (3.5 lakh parameters)", 
         "Sequential Neural Network 2 (8.4 lakh parameters)", "Sequential Neural Network 3 (8.1 lakh parameters)",
        "Sequential Neural Network 4 (26 thousand parameters)", "Sequential Neural Network 5 (6 thousand parameters)",
        "Convolutional Neural Network (Shallow)", "Convolutional Neural Network (Deep)")
    )

    predict_btn = st.button("Predict Digit", type="primary")


# --- 6. PROCESS PREDICTION IMMEDIATELY AFTER BUTTON CLICK ---
if predict_btn and canvas_result.image_data is not None:
    img_array = canvas_result.image_data
    gray_image = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)
    resized_image = cv2.resize(gray_image, (28, 28), interpolation=cv2.INTER_AREA)
    
    input_data = resized_image.reshape(1, 784) / 255.0
    
    st.session_state.current_input_data = input_data
    st.session_state.current_model_choice = model_choice
    st.session_state.prediction_made = True

    if model_choice == "Random Forest":
        st.session_state.current_prediction = rf_model.predict(input_data)[0]
    elif model_choice == "L1- Lasso Model":
        st.session_state.current_prediction = l1_model.predict(input_data)[0]
    elif model_choice == "Sequential Neural Network 1 (3.5 lakh parameters)":
        probabilities = nn_model_1.predict(input_data.reshape(1,28,28))[0]
        st.session_state.current_prediction = np.argmax(probabilities)
        st.session_state.probabilities = probabilities
    elif model_choice == "Sequential Neural Network 2 (8.4 lakh parameters)":
        probabilities = nn_model_2.predict(input_data.reshape(1,28,28))[0]
        st.session_state.current_prediction = np.argmax(probabilities)
        st.session_state.probabilities = probabilities
    elif model_choice == "Sequential Neural Network 3 (8.1 lakh parameters)":
        probabilities = nn_model_3.predict(input_data.reshape(1,28,28))[0]
        st.session_state.current_prediction = np.argmax(probabilities)
        st.session_state.probabilities = probabilities
    elif model_choice == "Sequential Neural Network 4 (26 thousand parameters)":
        probabilities = nn_model_4.predict(input_data.reshape(1,28,28))[0]
        st.session_state.current_prediction = np.argmax(probabilities)
        st.session_state.probabilities = probabilities
    elif model_choice == "Sequential Neural Network 5 (6 thousand parameters)":
        probabilities = nn_model_5.predict(input_data.reshape(1,28,28))[0]
        st.session_state.current_prediction = np.argmax(probabilities)
        st.session_state.probabilities = probabilities
    elif model_choice == "Convolutional Neural Network (Shallow)":
        probabilities = cnn_model_1.predict(input_data.reshape(1, 28, 28, 1))[0]
        st.session_state.current_prediction = np.argmax(probabilities)
        st.session_state.probabilities = probabilities
    elif model_choice == "Convolutional Neural Network (Deep)":
        probabilities = cnn_model_2.predict(input_data.reshape(1, 28, 28, 1))[0]
        st.session_state.current_prediction = np.argmax(probabilities)
        st.session_state.probabilities = probabilities


# --- 7. UI Layout: Feedback Widget (Left Column, Under Predict Button) ---
with col1:
    if st.session_state.prediction_made:
        st.divider()
        st.subheader("Hello! Please Help the models improve (Feedback Time!)")
        st.write("Did your drawing match with the number predited by the model?")

        col_fb1, col_fb2 = st.columns([1, 1])

        with col_fb1:
            is_correct = st.radio("Y/N", ("Yes", "No"), index=0, key="feedback_radio")

        if is_correct == "No":
            with col_fb2:
                actual_digit = st.number_input("What digit did you draw?", min_value=0, max_value=9, step=1)
                if st.button("Submit"):
                    success = log_to_google_sheet(
                        sheet_name="MNIST_Feedback_Data",  
                        model_used=st.session_state.current_model_choice,
                        predicted_label=st.session_state.current_prediction,
                        actual_label=actual_digit,
                        pixel_array=st.session_state.current_input_data,
                    )
                    if success:
                        st.success("Logged! Thank you for your valuable feedback")
                        time.sleep(1.2) 
                        st.session_state.prediction_made = False 
                        st.session_state.canvas_key += 1 
                        st.rerun() 

        elif is_correct == "Yes":
            with col_fb2:
                if st.button("Logged Sample"):
                    success = log_to_google_sheet(
                        sheet_name="MNIST_Feedback_Data",
                        model_used=st.session_state.current_model_choice,
                        predicted_label=st.session_state.current_prediction,
                        actual_label=st.session_state.current_prediction,
                        pixel_array=st.session_state.current_input_data,
                    )
                    if success:
                        st.success("Logged! Thanks for confirming.")
                        time.sleep(1.2) 
                        st.session_state.prediction_made = False 
                        st.session_state.canvas_key += 1 
                        st.rerun() 


# --- 8. UI Layout: Prediction Results (Right Column) ---
with col2:
    st.markdown("### Prediction Results")

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

        elif model_choice == "Sequential Neural Network 1 (3.5 lakh parameters)":
            probabilities = st.session_state.probabilities
            st.success(f"### Sequential Neural Network 1 (3.5 lakh params) predicts: **{prediction}**")
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

        elif model_choice == "Sequential Neural Network 2 (8.4 lakh parameters)":
            probabilities = st.session_state.probabilities
            st.success(f"### Sequential Neural Network 2 (8.8 lakh params) predicts: **{prediction}**")
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
                    
        elif model_choice == "Sequential Neural Network 3 (8.1 lakh parameters)":
            probabilities = st.session_state.probabilities
            st.success(f"### Sequential Neural Network 3 (8.1 lakh params) predicts: **{prediction}**")
            st.write(f"Confidence: {probabilities[prediction]:.2%}")
            st.progress(float(probabilities[prediction]))
            
            st.markdown("### Inside the Network: Neuron Activations")
            x = input_data.reshape(1, 28, 28)
            for i, layer in enumerate(nn_model_3.layers):
                x = layer(x)
                activation_data = x.numpy()
                st.caption(f"Layer {i+1}: {layer.name} ({activation_data.shape[-1]} neurons)")
                if len(activation_data.shape) == 2:
                    st.bar_chart(activation_data[0])

        elif model_choice == "Sequential Neural Network 4 (26 thousand parameters)":
            probabilities = st.session_state.probabilities
            st.success(f"### Sequential Neural Network 4 (26 thousand parameters) predicts: **{prediction}**")
            st.write(f"Confidence: {probabilities[prediction]:.2%}")
            st.progress(float(probabilities[prediction]))
            
            st.markdown("### Inside the Network: Neuron Activations")
            x = input_data.reshape(1, 28, 28)
            for i, layer in enumerate(nn_model_4.layers):
                x = layer(x)
                activation_data = x.numpy()
                st.caption(f"Layer {i+1}: {layer.name} ({activation_data.shape[-1]} neurons)")
                if len(activation_data.shape) == 2:
                    st.bar_chart(activation_data[0])

        elif model_choice == "Sequential Neural Network 5 (6 thousand parameters)":
            probabilities = st.session_state.probabilities
            st.success(f"### Sequential Neural Network 5 (6 thousand parameters) predicts: **{prediction}**")
            st.write(f"Confidence: {probabilities[prediction]:.2%}")
            st.progress(float(probabilities[prediction]))
            
            st.markdown("### Inside the Network: Neuron Activations")
            x = input_data.reshape(1, 28, 28)
            for i, layer in enumerate(nn_model_5.layers):
                x = layer(x)
                activation_data = x.numpy()
                st.caption(f"Layer {i+1}: {layer.name} ({activation_data.shape[-1]} neurons)")
                if len(activation_data.shape) == 2:
                    st.bar_chart(activation_data[0])

        elif model_choice == "Convolutional Neural Network (Shallow)":
            probabilities = st.session_state.probabilities
            st.success(f"### Convolutional Neural Network (Shallow) predicts: **{prediction}**")
            st.write(f"Confidence: {probabilities[prediction]:.2%}")
            st.progress(float(probabilities[prediction]))
            
            st.markdown("### Through the Eyes of the CNN")
            st.write("Here are the feature maps the first Convolutional layer extracted from your drawing:")
            
            # --- 1. FEATURE MAP VISUALIZATION ---
            conv_layer = None
            for layer in cnn_model_1.layers:
                if 'conv2d' in layer.name.lower():
                    conv_layer = layer
                    break
            
            if conv_layer:
                layer_output_model = Model(inputs=cnn_model_1.inputs, outputs=conv_layer.output)
                feature_maps = layer_output_model.predict(input_data.reshape(1, 28, 28, 1))
                num_filters = feature_maps.shape[-1]
                
                cols = 8
                rows = (num_filters // cols) + (1 if num_filters % cols != 0 else 0)
                fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
                
                for i, ax in enumerate(axes.flat):
                    if i < num_filters:
                        img = feature_maps[0, :, :, i]
                        ax.imshow(img, cmap='viridis')
                    ax.axis('off')
                    
                st.pyplot(fig)
            else:
                st.warning("Could not find a Convolutional layer to visualize in the Shallow model.")

            # --- 2. DENSE LAYER VISUALIZATION ---
            st.markdown("### Inside the Network: Dense Neuron Activations")
            st.write("After the features are extracted and pooled, they are flattened to make the final decision:")
            x = input_data.reshape(1, 28, 28, 1) # Note the 4D shape for CNNs
            for i, layer in enumerate(cnn_model_1.layers):
                x = layer(x)
                # Only plot layers that have been flattened (Flatten, Dense, Dropout)
                if len(x.shape) == 2:
                    activation_data = x.numpy()
                    st.caption(f"Layer {i+1}: {layer.name} ({activation_data.shape[-1]} neurons)")
                    st.bar_chart(activation_data[0])


        elif model_choice == "Convolutional Neural Network (Deep)":
            probabilities = st.session_state.probabilities
            st.success(f"### Convolutional Neural Network (Deep) predicts: **{prediction}**")
            st.write(f"Confidence: {probabilities[prediction]:.2%}")
            st.progress(float(probabilities[prediction]))
            
            st.markdown("### Through the Eyes of the Deep CNN")
            st.write("Watch how the network breaks down your drawing from basic edges to abstract concepts:")
            
            # --- 1. FEATURE MAP VISUALIZATION ---
            conv_layers = [layer for layer in cnn_model_2.layers if 'conv2d' in layer.name.lower()]
            
            if len(conv_layers) == 0:
                st.warning("Could not find any Convolutional layers to visualize in the Deep model.")
            else:
                for layer_idx, conv_layer in enumerate(conv_layers[:2]):
                    st.markdown(f"#### Layer {layer_idx + 1}: {conv_layer.name}")
                    if layer_idx == 0:
                        st.caption("Notice how these filters act like basic edge and curve detectors. The drawing is still very recognizable.")
                    else:
                        st.caption("Here, the filters are combining the edges from Layer 1 into more abstract, complex shapes. It looks blurrier because it's looking for concepts, not just lines.")
                    
                    layer_output_model = Model(inputs=cnn_model_2.inputs, outputs=conv_layer.output)
                    feature_maps = layer_output_model.predict(input_data.reshape(1, 28, 28, 1))
                    num_filters = feature_maps.shape[-1]
                    
                    cols = 8
                    rows = (num_filters // cols) + (1 if num_filters % cols != 0 else 0)
                    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
                    
                    for i, ax in enumerate(axes.flat):
                        if i < num_filters:
                            img = feature_maps[0, :, :, i]
                            ax.imshow(img, cmap='viridis')
                        ax.axis('off')
                        
                    st.pyplot(fig)

            # --- 2. DENSE LAYER VISUALIZATION ---
            st.markdown("### Inside the Network: Dense Neuron Activations")
            st.write("After the features are extracted and pooled, they are flattened to make the final decision:")
            x = input_data.reshape(1, 28, 28, 1) # Note the 4D shape for CNNs
            for i, layer in enumerate(cnn_model_2.layers):
                x = layer(x)
                # Only plot layers that have been flattened (Flatten, Dense, Dropout)
                if len(x.shape) == 2:
                    activation_data = x.numpy()
                    st.caption(f"Layer {i+1}: {layer.name} ({activation_data.shape[-1]} neurons)")
                    st.bar_chart(activation_data[0])
