from matplotlib import pyplot as plt
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
import os
from os import listdir
from numpy import save
from shutil import copyfile
from random import seed, random

data_folder = 'IMG_CLASSES/'
database_home = 'dataset/'

def data_processing():
    if not os.path.exists(database_home):
        os.makedirs(database_home)

    photos, labels = list(), list()
    
    # Class labels based on folder names
    class_mapping = {
        '1. Eczema': 0,
        '2. Melanoma': 1,
        '3. Atopic Dermatitis': 2,
        '4. Basel Cell Carcinoma': 3,
        '5. Melanocytic Nevi': 4,
        '6. Benign Keratosis-like Lesions (BKL)': 5,
        '7. Psoriasis Lichen Planus': 6,
        '8. Seborrheic Keratoses': 7,
        '9. Tinea Ringworm Candidiasis': 8,
        '10. Warts Molluscum': 9
    }

    # Load images and labels
    for folder_name in listdir(data_folder):
        folder_path = os.path.join(data_folder, folder_name)
        if os.path.isdir(folder_path):
            class_label = class_mapping.get(folder_name.split()[0] + " " + folder_name.split()[1], None)
            if class_label is not None:
                for filename in listdir(folder_path):
                    image_path = os.path.join(folder_path, filename)
                    photo = load_img(image_path, target_size=(224, 224))
                    photo = img_to_array(photo)
                    photos.append(photo)
                    labels.append(class_label)

    # Convert lists to numpy arrays
    photos = np.asarray(photos)
    labels = np.asarray(labels)

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
    
    # Copy data_folder images into train/test split based on val_ratio
    for folder in listdir(data_folder):
        folder_path = os.path.join(data_folder, folder)
        print(f'Checking folder: {folder_path}')
        if os.path.isdir(folder_path):
            print(f'Processing: {folder}')
            # Determine class name based on folder name
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
            else:
                continue

            # Process each image in the folder
            for filename in listdir(folder_path):
                src = os.path.join(folder_path, filename)
                # Assign to test or train set based on val_ratio
                dst_dir = 'train/' if random() >= val_ratio else 'test/'
                dst = os.path.join(database_home, dst_dir, class_name, filename)

                print(f"Copying {src} to {dst}")
                try:
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    copyfile(src, dst)
                except Exception as e:
                    print(f"Failed to copy {src} to {dst}: {e}")

data_processing()
insert_data()