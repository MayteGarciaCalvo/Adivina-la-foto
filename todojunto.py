from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model

import streamlit as st
from PIL import Image

import requests
import zipfile
import os

# URL de los datos
url = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"

# Descargar los datos
response = requests.get(url)
zip_path = "kagglecatsanddogs_5340.zip"

# Guardar el archivo zip
with open(zip_path, "wb") as f:
    f.write(response.content)

print(f"Archivo descargado en: {zip_path}")

# Descomprimir el archivo
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall()
    print("Archivos descomprimidos.")

# Comprobar que los datos se han descomprimido correctamente
base_dir = "PetImages"
if os.path.isdir(base_dir):
    print("Se han creado las subcarpetas correctamente en: PetImages")
    print(os.listdir(base_dir))  # Lista las subcarpetas
else:
    print("Error: No se encontraron las subcarpetas 'PetImages'.")

#Importamos las librerías necesarias

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

#Eliminar datos corruptos
num_skipped = 0
for folder_name in ("Cat", "Dog"):
    folder_path = os.path.join("PetImages", folder_name)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            with open(fpath, "rb") as fobj:  #with open() para cerrar automáticamente el archivo.
                is_jfif = tf.compat.as_bytes("JFIF") in fobj.read(10)
            if not is_jfif:
                num_skipped += 1
                os.remove(fpath)
        except Exception as e:
            print(f"⚠️ Error con {fpath}: {e}")
            num_skipped += 1
            os.remove(fpath)

print("Deleted %d images" % num_skipped)

# Definir directorios
data_dir = "PetImages"
train_dir = os.path.join(data_dir, "train")
validation_dir = os.path.join(data_dir, "validation")

# Cargar MobileNetV2 sin la capa de clasificación
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(160, 160, 3))
base_model.trainable = False  # Congelamos los pesos preentrenados

# Agregar capas personalizadas
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

#Este código confirma que las carpetas necesarias están creadas
base_dir = "PetImages"
sub_dirs = ["train", "validation", "test"]

for sub in sub_dirs:
    path = os.path.join(base_dir, sub)
    os.makedirs(path, exist_ok=True)

print("Estructura de carpetas creada correctamente.")

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

from depurar_y_confirmar_carpetas import base_model, predictions, data_dir, train_dir, validation_dir




#Este código distribuye las imágenes en train, validation y test:
base_dir = "PetImages"
categories = ["Cat", "Dog"]
split_ratio = [0.7, 0.15, 0.15]  # 70% train, 15% val, 15% test

for category in categories:
    src_folder = os.path.join(base_dir, category)
    images = [f for f in os.listdir(src_folder) if f.endswith(("jpg", "png", "jpeg"))]
    random.shuffle(images)

    train_split = int(len(images) * split_ratio[0])
    val_split = int(len(images) * (split_ratio[0] + split_ratio[1]))

    for i, img in enumerate(images):
        src_path = os.path.join(src_folder, img)
        if i < train_split:
            dst_folder = os.path.join(base_dir, "train", category.lower())
        elif i < val_split:
            dst_folder = os.path.join(base_dir, "validation", category.lower())
        else:
            dst_folder = os.path.join(base_dir, "test", category.lower())

        os.makedirs(dst_folder, exist_ok=True)
        shutil.move(src_path, os.path.join(dst_folder, img))

print("Imágenes organizadas correctamente en train, validation y test.")

# Crear el modelo
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Generadores de imágenes
train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(160, 160), batch_size=32, class_mode='binary')
validation_generator = val_datagen.flow_from_directory(
    validation_dir, target_size=(160, 160), batch_size=32, class_mode='binary')

# Entrenar el modelo
epochs = 5
model.fit(train_generator, validation_data=validation_generator, epochs=epochs)

# Guardar el modelo
model.save("cat_dog_classifier.h5")

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

from trainer import model

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