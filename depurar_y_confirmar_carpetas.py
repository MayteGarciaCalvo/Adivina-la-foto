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