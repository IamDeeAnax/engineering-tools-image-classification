import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import tensorflow_hub as hub
import base64

# Define a function to load the model with custom objects
def load_model_with_custom_objects(model_path):
    custom_objects = {"KerasLayer": hub.KerasLayer}
    return tf.keras.models.load_model(model_path, custom_objects=custom_objects)

# Load the best model
try:
    model = load_model_with_custom_objects('./model/engineering_tools_image_classification_model.keras')
    background_image_path = './assets/background.png'
    
    def add_bg_from_local(image_file):
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
            background-size: cover;
        }}
        .stApp h1 {{
            color: black;  /* Title color */
        }}
        .stApp .stMarkdown {{
            color: black;  /* Markdown text color */
        }}
        .stApp .stButton>button {{
            color: white;  /* Button text color */
        }}
        .stApp .stError {{
            color: red;  /* Error message color */
        }}

        .heading {{
            color: black;
            font-size: 35px;
            text-align: center;
            font-weight: bold;
        }}

        .success-message {{
            color: green;
            font-weight: bold;
            font-size: 20px;
        }}
        .warning-message {{
            color: yellow;
            font-weight: bold;
            font-size: 20px;
        }}

        .message {{
            color: black;
            font-weight: bold;
            font-size: 20px;
        }}

        .file-container {{
            text-align: center;
            margin-top: 20px;
        }}
        </style>
        """,
        unsafe_allow_html=True
        )
    add_bg_from_local(background_image_path)
    
    st.markdown("<h1 class='heading'>Engineering Tools Image Classification</h1>", unsafe_allow_html=True)

    # Determine the input shape of the model
    input_shape = model.input_shape[1:3]  # e.g., (224, 224) or (299, 299)

    # Initialize session state for the uploaded image and prediction result
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None
    if 'error_message' not in st.session_state:
        st.session_state.error_message = None

    # Upload image
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        st.session_state.error_message = None

    if st.session_state.uploaded_file is not None:
        img = Image.open(st.session_state.uploaded_file)
        st.image(img, caption='Uploaded Image.', use_column_width=True)

    # Add buttons to predict and clear image
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Predict Image"):
            if st.session_state.uploaded_file is not None:
                try:
                    img = Image.open(st.session_state.uploaded_file)
                    img = img.resize(input_shape)
                    img_array = image.img_to_array(img)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array /= 255.0

                    prediction = model.predict(img_array)
                    confidence_scores = tf.nn.softmax(prediction[0])
                    max_confidence = np.max(confidence_scores)
                    predicted_class = np.argmax(confidence_scores)

                    class_names = ['chisel', 'hammer', 'level', 'measuring_tape', 'plier', 'saw', 'screw_driver', 'wrench']

                    confidence_threshold = 0.7

                    if max_confidence > confidence_threshold:
                        st.session_state.prediction_result = f"The image is classified as: {class_names[predicted_class]} with confidence {max_confidence:.2f}"
                    else:
                        st.session_state.prediction_result = "The model is not confident enough to classify this image as any known category."

                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
            else:
                st.session_state.error_message = "Please upload an image to predict."

    with col2:
        if st.button("Clear Image"):
            st.session_state.uploaded_file = None
            st.session_state.prediction_result = None
            st.session_state.error_message = None
            st.experimental_rerun()

    # Display prediction result or error message
    if st.session_state.prediction_result:
        st.markdown(f"<p class='success-message'>{st.session_state.prediction_result}</p>", unsafe_allow_html=True)
    elif st.session_state.error_message:
        st.markdown(f"<p class='warning-message'>{st.session_state.error_message}</p>", unsafe_allow_html=True)
    elif st.session_state.uploaded_file is None:
        st.markdown(f"<p class='message'>Please upload an image to get started.</p>", unsafe_allow_html=True)

except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")