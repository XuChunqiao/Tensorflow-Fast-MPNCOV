from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow.keras.layers as layers
from src.representation.MPNCOV import *
import scipy.io as sio

__all__ = ['mpncov_vgg16bn']

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5
def batch_norm(init_zero=False, name=None):
    if init_zero:
        gamma_initializer = tf.zeros_initializer()
    else:
        gamma_initializer = tf.ones_initializer()

    if tf.keras.backend.image_data_format() == 'channels_last':
        axis = 3
    else:
        axis = 1
    return layers.BatchNormalization(axis=axis,
                                        momentum=BATCH_NORM_DECAY,
                                        epsilon=BATCH_NORM_EPSILON,
                                        center=True,
                                        scale=True,
                                        fused=True,
                                        gamma_initializer=gamma_initializer, name=name)
class MPNCOV_VGG(tf.keras.Model):
    def __init__(self, features, classes=1000):
        super(MPNCOV_VGG, self).__init__()
        self.features = features
        self.representation = MPNCOV(input_dim=512, dimension_reduction=256, iterNum=5)
        self.classifier = tf.keras.Sequential(
            layers=[
                layers.Flatten(),
                layers.Dense(4096, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(4096, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(classes)],
            name='classifier')
    def call(self, x, training=None):
        x = self.features(x, training=training)
        x = self.representation(x, training=training)
        x = self.classifier(x, training=training)
        return x

def mpncov_vgg16bn(pretrained):
    features = tf.keras.Sequential(
        layers=[
            # Block 1
            layers.Conv2D(64, (3, 3), padding="same", name='block1_conv1'),
            batch_norm(name='block1_bn1'),
            layers.ReLU(name='block1_relu1'),
            layers.Conv2D(64, (3, 3), padding="same", name='block1_conv2'),
            batch_norm(name='block1_bn2'),
            layers.ReLU(name='block1_relu2'),
            layers.MaxPooling2D((2, 2), (2, 2), name='block1_pool'),
            # Block 2
            layers.Conv2D(128, (3, 3), padding="same", name='block2_conv1'),
            batch_norm(name='block2_bn1'),
            layers.ReLU(name='block2_relu1'),
            layers.Conv2D(128, (3, 3), padding="same", name='block2_conv2'),
            batch_norm(name='block2_bn2'),
            layers.ReLU(name='block2_relu2'),
            layers.MaxPooling2D((2, 2), (2, 2), name='block2_pool'),
            # Block 3
            layers.Conv2D(256, (3, 3), padding="same", name='block3_conv1'),
            batch_norm(name='block3_bn1'),
            layers.ReLU(name='block3_relu1'),
            layers.Conv2D(256, (3, 3), padding="same", name='block3_conv2'),
            batch_norm( name='block3_bn2'),
            layers.ReLU(name='block3_relu2'),
            layers.Conv2D(256, (3, 3), padding="same", name='block3_conv3'),
            batch_norm(name='block3_bn3'),
            layers.ReLU(name='block3_relu3'),
            layers.MaxPooling2D((2, 2), (2, 2), name='block3_pool'),
            # Block 4
            layers.Conv2D(512, (3, 3), padding="same", name='block4_conv1'),
            batch_norm(name='block4_bn1'),
            layers.ReLU(name='block4_relu1'),
            layers.Conv2D(512, (3, 3), padding="same", name='block4_conv2'),
            batch_norm(name='block4_bn2'),
            layers.ReLU(name='block4_relu2'),
            layers.Conv2D(512, (3, 3), padding="same", name='block4_conv3'),
            batch_norm(name='block4_bn3'),
            layers.ReLU(name='block4_relu3'),
            layers.MaxPooling2D((2, 2), (2, 2), name='block4_pool'),
            # Block 5
            layers.Conv2D(512, (3, 3), padding="same", name='block5_conv1'),
            batch_norm(name='block5_bn1'),
            layers.ReLU(name='block5_relu1'),
            layers.Conv2D(512, (3, 3), padding="same", name='block5_conv2'),
            batch_norm(name='block5_bn2'),
            layers.ReLU(name='block5_relu2'),
            layers.Conv2D(512, (3, 3), padding="same", name='block5_conv3'),
            batch_norm(name='block5_bn3'),
            layers.ReLU(name='block5_relu3')],
        name='vgg16bn_features')
    model = MPNCOV_VGG(features)
    if pretrained:
        input = tf.random.normal([1, 224, 224, 3])
        model(input, training=False)
        # model.features.summary()
        # model.classifier.summary()
        weight_path = '/media/xcq/xcqdisk/MPNCOV_tensorflow/src/network/mpncov_vgg16bn_tf.h5'
        model.load_weights(weight_path)

    return model
