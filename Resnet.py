import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Add, GlobalAveragePooling2D, Dense, MaxPooling2D, Dropout, Flatten, Layer
from tensorflow.keras.models import Model, Sequential

class Identity(Layer):
    def __init__(self, filters, **kwargs):
        super(Identity, self).__init__(**kwargs)
        filter1, filter2, filter3 = filters
        self.conv1 = Conv2D(filter1, (1,1))
        self.bn1 = BatchNormalization(momentum=0.9)
        self.act1 = Activation('relu')
        self.conv2 = Conv2D(filter2, (3,3), padding='same')
        self.bn2 = BatchNormalization(momentum=0.9)
        self.act2 = Activation('relu')
        self.conv3 = Conv2D(filter3, (1,1))
        self.bn3 = BatchNormalization(momentum=0.9)
    def call(self, input_tensor, training=False):
        #layer1
        x = self.conv1(input_tensor)
        x = self.bn1(x, training=training)
        x = self.act1(x)
        #layer2
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.act2(x)
        #layer3
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = Add()([x, input_tensor])
        x = Activation('relu')(x)
        return x

class Convolutional(Layer):
    def __init__(self, filters, strides=(2,2), **kwargs):
        super(Convolutional, self).__init__(**kwargs)
        filter1, filter2, filter3 = filters
        self.conv1 = Conv2D(filter1, (1,1), strides=strides)
        self.bn1 = BatchNormalization(momentum=0.9)
        self.act1 = Activation('relu')
        self.conv2 = Conv2D(filter2, (3,3), padding='same')
        self.bn2 = BatchNormalization(momentum=0.9)
        self.act2 = Activation('relu')
        self.conv3 = Conv2D(filter3, (1,1))
        self.bn3 = BatchNormalization(momentum=0.9)
        self.shortcut_conv = Conv2D(filter3, (1,1), strides=strides)
        self.shortcut_bn = BatchNormalization(momentum=0.9)
    def call(self, input_tensor, training=False):
        #layer1
        x = self.conv1(input_tensor)
        x = self.bn1(x, training=training)
        x = self.act1(x)
        #layer2
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.act2(x)
        #layer3
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        #shortcut layer
        shortcut = self.shortcut_conv(input_tensor)
        shortcut = self.shortcut_bn(shortcut, training=training)
        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        return x
    

class Resnet(Model):
    def __init__(self, output_intermediate=False, **kwargs):
        super(Resnet, self).__init__(**kwargs)
        self.output_intermediate = output_intermediate
        #layer 1
        self.conv1 = Conv2D(64, (7, 7), strides=(2, 2), padding='same', input_shape=(224, 224, 3))
        self.bn1 = BatchNormalization(momentum=0.9)
        self.act1 = Activation('relu')
        self.maxpool1 = MaxPooling2D((3, 3), strides=(2,2), padding='same')
        self.res_block1 = self.resnet_block((64, 64, 256), 3, strides=(1, 1))
        self.res_block2 = self.resnet_block((128, 128, 512), 4, strides=(2, 2))
        self.res_block3 = self.resnet_block((256, 256, 1024), 6, strides=(2, 2))
        self.res_block4 = self.resnet_block((512, 512, 2048), 3, strides=(2, 2))
        self.avgpool = GlobalAveragePooling2D()
        self.flatten = Flatten()
        self.dense1 = Dense(512, activation='relu', kernel_initializer='he_uniform')
        self.dropout1 = Dropout(0.5)
        self.dense2 = Dense(128, activation='relu')
        self.dropout2 = Dropout(0.3)
        self.dense3 = Dense(10, activation='softmax')
    def resnet_block(self, filters, blocks, strides=(2,2)):
        res_blocks = []
        res_blocks.append(Convolutional(filters, strides))
        for _ in range(1, blocks):
            res_blocks.append(Identity(filters))
        return Sequential(res_blocks)
    def call(self, x, training=False):
        #Stage 1
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.act1(x)
        x = self.maxpool1(x)
        #Stage 2
        x = self.res_block1(x, training=training)
        #stage 3
        x = self.res_block2(x, training=training)
        #Stage 4
        x = self.res_block3(x, training=training)
        #Stage 5
        x = self.res_block4(x, training=training)
        if self.output_intermediate:
            return x
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        return self.dense3(x)



        
