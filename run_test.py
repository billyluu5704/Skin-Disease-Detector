from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
# Limit TensorFlow to use only a fraction of the GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input, Dropout, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import concatenate, BatchNormalization, Activation
from tensorflow.keras.models import Model, Sequential, load_model, save_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import SGD, Adam
from keras.preprocessing.image import load_img, img_to_array
from numpy import mean, std
import os
from os import listdir
from numpy import save
from keras.models import load_model
from Resnet import Resnet
from PIL import Image
import tkinter as tk
from tkinter import filedialog

model_file = 'train_model.keras'
model_name = Resnet()

def recreate_model():
    return Resnet()

def load_image(filename):
    img = load_img(filename, target_size=(224, 224))
    img = img_to_array(img)
    img = img.reshape(1, 224, 224, 3)
    img = img.astype('float32')
    img -= [123.68, 116.779, 103.939]
    return img

def open_image():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")]
    )
    return file_path

def run_model():
    file = open_image()
    if file:
        print(f'File: {file} selected')
        model = load_model(model_file, custom_objects={'Resnet': recreate_model()})
        img = load_image(file)
        pred = model.predict(img)
        result = np.argmax(pred, axis=1)
        return result[0]
    else:
        print('No file selected')


result = run_model()
diseases = ['Eczema', 'Melanoma', 'Atopic Dermatitis', 'Basel Cell Carcinoma',
                    'Melanocytic Nevi', 'Benign Keratosis-like Lesions (BKL)', 
                    'Psoriasis Lichen Planus', 'Seborrheic Keratoses', 
                    'Tinea Ringworm Candidiasis', 'Warts Molluscum']
if result == 0.0:
    print(f'You need to make an appointment with a doctor because you have a high chance of having {diseases[0]} and need cortocosteroids and you should buy treatment cream')
elif result == 1.0:
    print(f'This is {diseases[1]} (which is skin cancer). Please visit the closest hospital to examine the sign appears on your skin before it gets worse!')
elif result == 2.0:
    print(f"This is {diseases[2]}, you may start with regular moisturizing and other self-care habits. If these don't help, your health care provider might suggest medicated creams that control itching and help repair skin. These are sometimes combined with other treatments.")
elif result == 3.0:
    print(f"This is {diseases[3]}, it is not life-threatening and does not spread, but can grow larger if left untreated. Please make an appointment with doctor and buy Hydroquinone Cream to have on your skin!")
elif result == 4.0:
    print(f"No worries, this is called {diseases[4]}. Please buy Imiquimod 5% Cream!")
elif result == 5.0:
    print(f"You got {diseases[5]}. It won’t require any treatment. But you might want to have it removed if it becomes itchy or irritated or you don’t like the look of it. Your healthcare provider can remove it for you in the office using one of several common methods.")
elif result == 6.0:
    print(f"You got {diseases[6]}. These following treatments may help relieve your symptoms: Corticosteroid creams, Antihistamines, Phototherapy, and Immunosuppressants.")
elif result == 7.0:
    print(f"You got {diseases[7]}. You should visit nearest medical office to receive the following treatments like: Cryotherapy, Electrodessication, Shave Excision, or Laser Therapy.")
elif result == 8.0:
    print(f"You got {diseases[8]}. You should buy some non-prescription like: Clotrimazole, Miconazole, Terbinafine, or Ketoconazole. If it got worse, please contact a healthcare provider!")
elif result == 9.0:
    print(f"You got {diseases[9]}. You should visit a healthcare provider for further examination. You should also buy the following unprescribed treatment: Imiquimod cream.")