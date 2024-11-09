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
import sys
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib.image import imread
from shutil import copyfile
from random import seed, random
from AlexNet import AlexNet
from VGG import VGG
from Resnet import Resnet

data_folder = 'IMG_CLASSES/'
database_home = 'dataset/'

def define_model():
    model = Resnet()
    opt = Adam()
    model.compile(optimizer=opt, loss='crossentropy', metrics=['accuracy'])
    return model

#plot diagnostic learning curves
def summarize_diagnostics(histories):
    #plot loss
    plt.subplot(1, 2, 1)
    plt.title('Cross Entropy Loss')
    plt.plot(histories.history['loss'], color='blue', label='train')
    plt.plot(histories.history['val_loss'], color='orange', label='test')
    #plot accuracy
    plt.subplot(1, 2, 2)
    plt.title('Classification Accuracy')
    plt.plot(histories.history['accuracy'], color='blue', label='train')
    plt.plot(histories.history['val_accuracy'], color='orange', label='test')
    plt.show()
    #save plot to file
    filename = sys.argv[0].split('/')[-1]
    plt.savefig(filename + '_plot.png')
    plt.close()

def run_test_harness():
    #data_processing()
    #insert_data()
    train_datagen = ImageDataGenerator(rescale=1.0/255.0, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1.0/255.0)
    train_it = train_datagen.flow_from_directory(f'{database_home}train/', class_mode='categorical', batch_size= 128, target_size=(224,224))
    test_it = test_datagen.flow_from_directory(f'{database_home}test/', class_mode='categorical', batch_size=128, target_size=(224,224))
    #load model
    filename = 'model.keras'
    if os.path.isfile(filename):
        model = load_model(filename)
        # Compile model with the desired optimizer and loss function
        opt = Adam()
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        #define model
        model = define_model()
    #fit model
    history = model.fit(train_it, 
                        steps_per_epoch=(train_it.samples // train_it.batch_size), 
                        validation_data=test_it, 
                        validation_steps=(test_it.samples // test_it.batch_size), 
                        epochs=50, verbose=1)
    #evaluate model
    _, acc = model.evaluate(test_it, 
                            steps=(test_it.samples // test_it.batch_size), 
                            verbose=0) 
    print('> %.3f' % (acc * 100.0))
    #learning curves
    summarize_diagnostics(history)
    #save model
    model.save(filename)

run_test_harness()