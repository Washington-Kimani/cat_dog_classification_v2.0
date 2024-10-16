import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Set page configuration
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐱🐶",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Load the model
model = load_model('/home/codename/projects/learning/ds n ml/models/cat_dog_classification_v2.0/models/dog_cat_classifier.h5')

# Set a stylish header
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Cat vs Dog Classifier 🐱🐶</h1>", unsafe_allow_html=True)

# Introduction text
st.write("Welcome! Upload an image, and our AI model will predict whether it's a **Cat** or a **Dog**.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.write("### File uploaded successfully! 📂")
    
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', width=300)
    st.write("### Image displayed successfully!")
    
    # Image preprocessing and prediction
    try:
        with st.spinner('Processing image... 🌀'):
            # Preprocess the image to fit the model input
            image = tf.image.resize(image, (256, 256))  
            image = np.array(image)
            
            # Model prediction
            y_pred = model.predict(np.expand_dims(image / 255, axis=0))
            y_pred_prob = y_pred[0][0]
            y_pred = y_pred_prob > 0.5
            
            # Show prediction result
            if y_pred == 0:
                pred = 'Cat 😺'
            else:
                pred = 'Dog 🐶'

            # Display result with confidence score
            st.success(f"Our model predicts: **{pred}** with a confidence of {y_pred_prob*100:.2f}%")
    except Exception as e:
        st.error("Error during image processing or prediction:")
        st.write(e)
else:
    st.warning("Please upload an image file to proceed.")
