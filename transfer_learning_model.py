from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('float32')
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

resnet_backbone = Resnet(output_intermediate=True)

class RPN(Model):
    def __init__(self, anchors=9, **kwargs):
        super(RPN, self).__init__(**kwargs)
        self.conv = layers.Conv2D(512, (3, 3), activation='relu', padding='same')
        self.cls_layer = layers.Conv2D(anchors, (1, 1), activation='sigmoid')
        self.reg_layer = layers.Conv2D(anchors * 4, (1, 1))
    def call(self, feature_map, training=False):
        x = self.conv(feature_map)
        objectness = self.cls_layer(x)
        bbox_deltas = self.reg_layer(x)
        return objectness, bbox_deltas
    
class DetectionHead(Model):
    def __init__(self, num_bboxes=10, num_classes=10, **kwargs):
        super(DetectionHead, self).__init__(**kwargs)
        self.num_bboxes = num_bboxes
        self.fc1 = layers.Dense(1024, activation='relu')
        self.fc2 = layers.Dense(1024, activation='relu')
        self.bbox_fc = layers.Dense(self.num_bboxes*4)  # Ensure the output matches the target shape
        self.cls_fc = layers.Dense(num_classes, activation='softmax')
        self.mask_deconv1 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), activation='relu')
        self.mask_out = layers.Conv2D(num_classes, (1, 1), activation='sigmoid')
        self.flatten_bbox = layers.Flatten()
    def call(self, x, training=False):
        x_flattened = self.flatten_bbox(x)
        class_logits = self.cls_fc(self.fc2(self.fc1(x_flattened)))

        # Bounding box coordinates - flatten to ensure it's compatible with loss
        bbox = self.bbox_fc(x_flattened)
        bbox = tf.reshape(bbox, [-1, self.num_bboxes, 4]) 

        # Mask prediction
        mask = self.mask_deconv1(x)
        mask = self.mask_out(mask)

        return class_logits, bbox, mask

@tf.function(jit_compile=False)
def roi_align(feature_map, rois, box_indices):
    return tf.image.crop_and_resize(feature_map, rois, box_indices, crop_size=(7, 7))

    
class MaskRCNN(Model):
    def __init__(self, backbone=resnet_backbone, num_classes=10, **kwargs):
        super(MaskRCNN, self).__init__(**kwargs)
        self.backbone = backbone
        self.rpn = RPN()
        self.detection_head = DetectionHead(num_classes=num_classes)
    def call(self, images, training=False):
        feature_map = self.backbone(images, training=training)
        objectness, bbox_deltas = self.rpn(feature_map, training=training)
        rois = tf.constant([[0, 0, 32, 32]], dtype=tf.float32)  # Example placeholder ROIs
        box_indices = tf.zeros((tf.shape(rois)[0],), dtype=tf.int32)  # Batch index for each ROI
        aligned_rois = roi_align(feature_map, rois, box_indices)
        class_logits, bbox, mask = self.detection_head(aligned_rois, training=training)
        return {
            'class_logits': class_logits,
            'bbox': bbox,
            'mask': mask
        }


