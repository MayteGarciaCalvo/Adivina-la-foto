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

from train import validation_generator, model, train_generator


# Evaluar el modelo con el conjunto de validación
score = model.evaluate(validation_generator, verbose=0)

# Mostrar resultados
print("Validation loss:", score[0])
print("Validation accuracy:", score[1])

history=model.fit(
    train_generator,
    steps_per_epoch= 100,
    epochs= 100,
    validation_data=validation_generator,
    validation_steps= 50
)

# visualizamos los resultados como antes
# almacenamos acc, loss, val_acc y val_loss desde history
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
# definimos epochs con rango 1 a len(acc)+1
epochs = range(1, len(acc)+1)
# creamos grafico con acc y val_acc
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()
# creamos gráfico con loss y val_loss
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
# mostramos gráfico
plt.show()

#Comprobar Overfitting
train_loss, train_accuracy = model.evaluate(train_generator, verbose=0)
val_loss, val_accuracy = model.evaluate(validation_generator, verbose=0)

print(f"Train Loss: {train_loss:.4f} - Train Accuracy: {train_accuracy:.4%}")
print(f"Validation Loss: {val_loss:.4f} - Validation Accuracy: {val_accuracy:.4%}")