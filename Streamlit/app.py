import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import tensorflow_hub as hub

# Define a function to load the model with custom objects
def load_model_with_custom_objects(model_path):
    custom_objects = {"KerasLayer": hub.KerasLayer}
    return tf.keras.models.load_model(model_path, custom_objects=custom_objects)

# Load the best model
try:
    model = load_model_with_custom_objects('model/engineering_tools_image_classification_model.keras')
    st.title('Engineering Tools Image Classification')

    # Determine the input shape of the model
    input_shape = model.input_shape[1:3]  # e.g., (224, 224) or (299, 299)

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            img = Image.open(uploaded_file)
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
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
