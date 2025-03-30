import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

import os,shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
import random

#import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

from train import model

# Cargar el modelo
model = tf.keras.models.load_model("cat_dog_classifier.h5")

# Función para predecir
def predict_image(image):
    image = image.resize((160, 160))  # Redimensionar a 160x160
    image_array = np.array(image) / 255.0  # Normalizar
    image_array = np.expand_dims(image_array, axis=0)  # Añadir batch
    prediction = model.predict(image_array)[0][0]  # Obtener predicción
        # Determinar si la predicción es confiable
    if prediction > 0.9:  # Umbral alto para una predicción confiable
        return "Perro 🐶"
    elif prediction < 0.1:  # Umbral bajo para una predicción confiable
        return "Gato 🐱"
    else:
        return "No se puede clasificar la imagen. El modelo solo reconoce perros y gatos."

# Interfaz de Streamlit
st.title("Clasificador de Perros y Gatos 🐶🐱")
st.write("Sube una imagen y el modelo la clasificará como gato o perro.")

uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen cargada", use_column_width=True)
    
# Realizar predicción
    prediction = predict_image(image)
    st.subheader(f"Resultado: {prediction}")