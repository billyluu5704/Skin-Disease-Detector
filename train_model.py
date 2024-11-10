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
from tensorflow.keras.models import Model, Sequential, load_model, save_model
from tensorflow.keras.optimizers import SGD, Adam
import os
from keras.models import load_model
import sys
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from Resnet import Resnet
from transfer_learning_model import MaskRCNN


data_folder = 'IMG_CLASSES/'
database_home = 'dataset/'

def define_model():
    model = Resnet()
    opt = Adam(learning_rate=0.0001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
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
    plt.title('Classification F1 Score')
    plt.plot(histories.history['accuracy'], color='blue', label='train')
    plt.plot(histories.history['val_accuracy'], color='orange', label='test')
    plt.show()
    #save plot to file
    filename = sys.argv[0].split('/')[-1]
    plt.savefig(filename + '_plot.png')
    plt.close()

def run_test_harness():
    train_datagen = ImageDataGenerator(rescale=1.0/255.0, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, rotation_range=20, zoom_range=0.2)
    test_datagen = ImageDataGenerator(rescale=1.0/255.0)
    train_it = train_datagen.flow_from_directory(f'{database_home}train/', class_mode='categorical', batch_size=32, target_size=(224,224))
    test_it = test_datagen.flow_from_directory(f'{database_home}test/', class_mode='categorical', batch_size=32, target_size=(224,224))
    #load model
    filename = 'train_model.keras'
    if os.path.isfile(filename):
        model = load_model(filename)
        opt = Adam(learning_rate=0.0001)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        #define model
        model = define_model()
    #fit model
    history = model.fit(train_it, 
                        steps_per_epoch=(train_it.samples // train_it.batch_size), 
                        validation_data=test_it, 
                        validation_steps=(test_it.samples // test_it.batch_size), 
                        epochs=80, verbose=1)
    #evaluate model
    _, acc = model.evaluate(test_it, 
                            steps=(test_it.samples // test_it.batch_size), 
                            verbose=0) 
    print(f'Accuracy: {acc:.3f}')
    #save model
    model.save(filename)
    #learning curves
    summarize_diagnostics(history)

run_test_harness()