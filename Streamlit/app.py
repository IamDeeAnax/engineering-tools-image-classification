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
    model = load_model_with_custom_objects('Streamlit/model/engineering_tools_image_classification_model.keras')
    background_image_path = 'assets/background.png'
    
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
            color: black;  /* Button text color */
        }}
        .stApp .stError {{
            color: red;  /* Error message color */
        }}

        </style>
        """,
        unsafe_allow_html=True
        )
    add_bg_from_local(background_image_path)
    
    st.markdown("<h1 class='heading'>Engineering Tools Image Classification</h1>", unsafe_allow_html=True)

    # Determine the input shape of the model
    input_shape = model.input_shape[1:3]  # e.g., (224, 224) or (299, 299)

    # Initialize session state for the uploaded image
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None

    # Upload image
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file

    if st.session_state.uploaded_file is not None:
        try:
            img = Image.open(st.session_state.uploaded_file)
            st.image(img, caption='Uploaded Image.', use_column_width=True)

            # Ensure image has three channels (RGB)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Preprocess the image
            img = img.resize(input_shape)  # Resize to the correct dimensions
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0

            # Predict the class of the image
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction, axis=-1)

            # Define the class names based on your model's training
            class_names = ['chisel', 'hammer', 'level', 'measuring_tape', 'plier', 'saw', 'screw_driver', 'wrench']

            # Display the prediction
            st.write(f"The image is classified as: {class_names[predicted_class[0]]}")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
    else:
        st.write("Please upload an image to get started.")
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")

