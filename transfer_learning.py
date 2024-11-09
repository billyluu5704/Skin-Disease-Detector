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
from Resnet import Resnet