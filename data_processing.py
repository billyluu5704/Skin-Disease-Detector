from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model, save_model
from tensorflow.keras.optimizers import SGD, Adam
from keras.preprocessing.image import load_img, img_to_array
import os
from os import listdir
from numpy import save
from keras.models import load_model
import sys
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib.image import imread
from shutil import copyfile
from random import seed, random
from Resnet import Resnet
import tkinter as tk
from tkinter import filedialog
from PIL import Image

data_folder = 'IMG_CLASSES/'
database_home = 'dataset/'

def data_processing():
    if not os.path.exists(database_home):
        os.makedirs(database_home)
    
    photos, labels = list(), list()
    
    for folder_name in listdir(data_folder):
        folder_path = os.path.join(data_folder, folder_name)
        if os.path.isdir(folder_path):  # Corrected to use folder_path
            if folder_name.startswith('1. Eczema'):
                output = 0.0
            elif folder_name.startswith('2. Melanoma'):
                output = 1.0
            elif folder_name.startswith('3. Atopic Dermatitis'):
                output = 2.0
            elif folder_name.startswith('4. Basel Cell Carcinoma'):
                output = 3.0
            elif folder_name.startswith('5. Melanocytic Nevi'):
                output = 4.0
            elif folder_name.startswith('6. Benign Keratosis-like Lesions (BKL)'):
                output = 5.0
            elif folder_name.startswith('7. Psoriasis Lichen Planus'):
                output = 6.0
            elif folder_name.startswith('8. Seborrheic Keratoses'):
                output = 7.0
            elif folder_name.startswith('9. Tinea Ringworm Candidiasis'):
                output = 8.0
            elif folder_name.startswith('10. Warts Molluscum'):
                output = 9.0
            
            for filename in listdir(folder_path):
                image_path = os.path.join(folder_path, filename)
                # Load photos
                photo = load_img(image_path, target_size=(224, 224))
                # Convert to numpy array
                photo = img_to_array(photo)
                photos.append(photo)
                labels.append(output)

    # Convert lists to numpy arrays after the loop
    photos = np.asarray(photos)
    labels = np.asarray(labels)
    print(photos.shape, labels.shape)
    
    # Save arrays
    save('photos.npy', photos)
    save('labels.npy', labels)
    
    subdirs = ['train/', 'test/']
    for subdir in subdirs:
        labeldirs = ['Eczema/', 'Melanoma/', 'Atopic Dermatitis/', 
                     'Basel Cell Carcinoma/', 'Melanocytic Nevi/', 
                     'Benign Keratosis-like Lesions (BKL)/', 'Psoriasis Lichen Planus/', 
                     'Seborrheic Keratoses/', 'Tinea Ringworm Candidiasis/', 'Warts Molluscum/']
        for labldir in labeldirs:
            newdir = os.path.join(database_home, subdir, labldir)
            if not os.path.exists(newdir):
                os.makedirs(newdir)

def insert_data():
    seed(1)
    val_ratio = 0.25
    #copy data_folder images
    for folder in listdir(data_folder):
        folder_path = os.path.join(data_folder, folder)
        print(f'Check folder: {folder_path}')
        if os.path.isdir(folder_path):
            print(f'Processing: {folder}')
            for filename in listdir(folder_path):
                src = os.path.join(folder_path, filename)
                dst_dir = 'train/'
                if random() < val_ratio:
                    dst_dir = 'test/'
                dst = None
                if 'Eczema' in folder:
                    class_name = 'Eczema'
                elif 'Melanoma' in folder:
                    class_name = 'Melanoma'
                elif 'Atopic Dermatitis' in folder:
                    class_name = 'Atopic Dermatitis'
                elif 'Basal Cell Carcinoma' in folder:
                    class_name = 'Basal Cell Carcinoma'
                elif 'Melanocytic Nevi' in folder:
                    class_name = 'Melanocytic Nevi'
                elif 'Benign Keratosis-like Lesions (BKL)' in folder:
                    class_name = 'Benign Keratosis-like Lesions (BKL)'
                elif 'Psoriasis Lichen Planus' in folder:
                    class_name = 'Psoriasis Lichen Planus'
                elif 'Seborrheic Keratoses' in folder:
                    class_name = 'Seborrheic Keratoses'
                elif 'Tinea Ringworm Candidiasis' in folder:
                    class_name = 'Tinea Ringworm Candidiasis'
                elif 'Warts Molluscum' in folder:
                    class_name = 'Warts Molluscum'              
                dst = os.path.join(database_home, dst_dir, class_name, filename)
                if dst:
                    print(f"Copying {src} to {dst}")
                    try:
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        copyfile(src, dst)
                    except Exception as e:
                        print(f"Failed to copy {src} to {dst}: {e}")