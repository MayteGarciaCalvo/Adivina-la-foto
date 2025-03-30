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

