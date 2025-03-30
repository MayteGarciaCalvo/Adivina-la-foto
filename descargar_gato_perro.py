
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