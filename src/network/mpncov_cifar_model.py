'''
resnet for cifar in pytorch

Reference:
[1] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.
[2] K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. In ECCV, 2016.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from ..representation.MPNCOV import *

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5

__all__ = ['MPNCOV_ResNet_Cifar', 'MPNCOV_PreAct_ResNet_Cifar', 'cifar_mpncovresnet20', 'cifar_mpncovresnet32',
           'cifar_mpncovresnet44', 'cifar_mpncovresnet56', 'cifar_mpncovresnet110', 'cifar_mpncovresnet1202',
           'cifar_mpncovresnet164', 'cifar_mpncovresnet1001', 'cifar_preact_mpncovresnet110',
           'cifar_preact_mpncovresnet164', 'cifar_preact_mpncovresnet1001']

def conv1X1(filters, stride=1):
    return layers.Conv2D(filters, kernel_size=1, strides=stride, use_bias=False, padding='same')

def conv3X3(filters, stride=1):
    " 3x3 convolution with padding "
    return tf.keras.Sequential([
        layers.ZeroPadding2D(padding=1),
        layers.Conv2D(filters, kernel_size=3, strides=stride, use_bias=False, padding='valid')])

def batch_norm(init_zero=False):
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
                                        gamma_initializer=gamma_initializer)
class downsample_block(tf.keras.Model):
    expansion = 1
    def __init__(self, filters, strides=1):
        super(downsample_block, self).__init__()
        self.downsample_conv = conv1X1(filters, strides)
        self.downsample_bn = batch_norm(init_zero=False)
    def call(self, x, training):
        out = self.downsample_conv(x)
        out = self.downsample_bn(out, training=training)
        return out

class residual_block(tf.keras.Model):
    expansion = 1
    def __init__(self, filters, strides=1, downsample=None):
        super(residual_block, self).__init__()
        self.conv1 = conv3X3(filters=filters, stride=strides)
        self.bn1 = batch_norm(init_zero=False)
        self.relu = layers.ReLU()
        self.conv2 = conv3X3(filters=filters)
        self.bn2 = batch_norm(init_zero=False)
        self.downsample = downsample
        self.strides = strides
    def call(self, x, training):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out, training=training)

        if self.downsample is not None:
            identity = self.downsample(x, training=training)

        out += identity
        out = self.relu(out)
        return out


class bottleneck_block(tf.keras.Model):
    expansion = 4

    def __init__(self, filters, strides=1, downsample=None):
        super(bottleneck_block, self).__init__()
        self.conv1 = conv1X1(filters=filters)
        self.bn1 = batch_norm(init_zero=False)
        self.conv2 = conv3X3(filters=filters, stride=strides)
        self.bn2 = batch_norm(init_zero=False)
        self.conv3 = conv1X1(filters=filters * self.expansion)
        self.bn3 = batch_norm(init_zero=False)
        self.relu = layers.ReLU()
        self.downsample = downsample
        self.strides = strides
    def call(self, x ,training):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = self.relu(out)


        out = self.conv2(out)
        out = self.bn2(out, training=training)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out, training=training)
        if self.downsample is not None:
            identity = self.downsample(x, training=training)
        out += identity
        out = self.relu(out)
        return out

class PreAct_residual_block(tf.keras.Model):
    expansion = 1
    def __init__(self, filters, strides=1, downsample=None):
        super(PreAct_residual_block, self).__init__()
        self.bn1 = batch_norm(init_zero=False)
        self.relu = layers.ReLU()
        self.conv1 = conv3X3(filters=filters, stride=strides)
        self.bn2 = batch_norm(init_zero=False)
        self.conv2 = conv3X3(filters=filters)
        self.downsample = downsample
        self.strides = strides
    def call(self, x, training):
        identity = x
        out = self.bn1(x, training=training)
        out = self.relu(out)

        if self.downsample is not None:
            identity = self.downsample(out)

        out = self.conv1(out)

        out = self.bn2(out, training=training)
        out = self.relu(out)
        out = self.conv2(out)

        out += identity

        return out
class PreAct_bottleneck_block(tf.keras.Model):
    expansion = 4

    def __init__(self, filters, strides=1, downsample=None):
        super(PreAct_bottleneck_block, self).__init__()

        self.bn1 = batch_norm(init_zero=False)
        self.relu = layers.ReLU()
        self.conv1 = conv1X1(filters=filters)

        self.bn2 = batch_norm(init_zero=False)
        self.conv2 = conv3X3(filters=filters, stride=strides)

        self.bn3 = batch_norm(init_zero=False)
        self.conv3 = conv1X1(filters=filters * self.expansion)

        self.downsample = downsample
        self.strides = strides
    def call(self, x ,training):
        identity = x

        out = self.bn1(x, training=training)
        out = self.relu(out)
        if self.downsample is not None:
            identity = self.downsample(out)

        out = self.conv1(out)

        out = self.bn2(out, training=training)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out, training=training)
        out = self.relu(out)
        out = self.conv3(out)

        out += identity

        return out
class MPNCOV_ResNet_Cifar(tf.keras.Model):
    def __init__(self, block_fn, blocks, num_classes=10):
        super(MPNCOV_ResNet_Cifar, self).__init__()
        self.in_filters = 16
        self.num_classes = num_classes
        self.conv1 = layers.Conv2D(filters=16, kernel_size=3, strides=1, use_bias=False, padding='same')
        self.bn1 = batch_norm(init_zero=False)
        self.relu = layers.ReLU()
        self.layer1 = self._make_layer(block_fn, 16, blocks[0], name='stage1')
        self.layer2 = self._make_layer(block_fn, 32, blocks[1], stride=2, name='stage2')
        self.layer3 = self._make_layer(block_fn, 64, blocks[2], stride=1, name='stage3')

        self.MPNCOV = MPNCOV(input_dim=256, iterNum=5, dimension_reduction=128, dropout_p=0.5)
        self.fc = layers.Dense(self.num_classes)


    def _make_layer(self, block, filters, blocks, stride=1, name=None):
        downsample = None
        if stride != 1 or self.in_filters != filters * block.expansion:
            downsample = downsample_block(filters=filters * block.expansion, strides=stride)

        block_layers= []
        block_layers.append(block(filters, stride, downsample))
        self.in_filters = filters * block.expansion
        for i in range(1, blocks):
            block_layers.append(block(filters))

        return tf.keras.Sequential(layers=block_layers, name=name)

    def call(self, x, training):
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.relu(x)

        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)

        x = self.MPNCOV(x, training=training)

        x = self.fc(x)
        return x


class MPNCOV_PreAct_ResNet_Cifar(tf.keras.Model):
    def __init__(self, block_fn, blocks, num_classes=100):
        super(MPNCOV_PreAct_ResNet_Cifar, self).__init__()
        self.in_filters = 16
        self.num_classes = num_classes
        self.conv1 = layers.Conv2D(filters=16, kernel_size=3, strides=1, use_bias=False, padding='same')

        self.layer1 = self._make_layer(block_fn, 16, blocks[0], name='stage1')
        self.layer2 = self._make_layer(block_fn, 32, blocks[1], stride=2, name='stage2')
        self.layer3 = self._make_layer(block_fn, 64, blocks[2], stride=1, name='stage3')

        self.bn = batch_norm(init_zero=False)
        self.relu = layers.ReLU()

        self.MPNCOV = MPNCOV(input_dim=256, iterNum=5, dimension_reduction=128, dropout_p=0.5)

        self.fc = layers.Dense(self.num_classes)

    def _make_layer(self, block, filters, blocks, stride=1, name=None):
        downsample = None
        if stride != 1 or self.in_filters != filters * block.expansion:
            downsample = conv1X1(filters=filters * block.expansion, stride=stride)

        block_layers = []
        block_layers.append(block(filters, stride, downsample))
        self.in_filters = filters * block.expansion
        for i in range(1, blocks):
            block_layers.append(block(filters))

        return tf.keras.Sequential(layers=block_layers, name=name)

    def call(self, x, training):
        x = self.conv1(x)

        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)

        x = self.bn(x, training=training)
        x = self.relu(x)

        x = self.MPNCOV(x, training=training)

        x = self.fc(x)
        return x

def cifar_mpncovresnet20(pretrained=False, **kwargs):
    model = MPNCOV_ResNet_Cifar(residual_block, [3, 3, 3], **kwargs)
    return model


def cifar_mpncovresnet32(pretrained=False, **kwargs):
    model = MPNCOV_ResNet_Cifar(residual_block, [5, 5, 5], **kwargs)
    return model


def cifar_mpncovresnet44(pretrained=False, **kwargs):
    model = MPNCOV_ResNet_Cifar(residual_block, [7, 7, 7], **kwargs)
    return model


def cifar_mpncovresnet56(pretrained=False, **kwargs):
    model = MPNCOV_ResNet_Cifar(residual_block, [9, 9, 9], **kwargs)
    return model


def cifar_mpncovresnet110(pretrained=False, **kwargs):
    model = MPNCOV_ResNet_Cifar(residual_block, [18, 18, 18], **kwargs)
    return model


def cifar_mpncovresnet1202(pretrained=False, **kwargs):
    model = MPNCOV_ResNet_Cifar(residual_block, [200, 200, 200], **kwargs)
    return model


def cifar_mpncovresnet164(pretrained=False, **kwargs):
    model = MPNCOV_ResNet_Cifar(bottleneck_block, [18, 18, 18], **kwargs)
    return model


def cifar_mpncovresnet1001(pretrained=False, **kwargs):
    model = MPNCOV_ResNet_Cifar(bottleneck_block, [111, 111, 111], **kwargs)

    return model


def cifar_preact_mpncovresnet110(pretrained=False, **kwargs):
    model = MPNCOV_PreAct_ResNet_Cifar(PreAct_residual_block, [18, 18, 18], **kwargs)
    return model


def cifar_preact_mpncovresnet164(pretrained=False, **kwargs):
    model = MPNCOV_PreAct_ResNet_Cifar(PreAct_bottleneck_block, [18, 18, 18], **kwargs)
    return model


def cifar_preact_mpncovresnet1001(pretrained=False, **kwargs):
    model = MPNCOV_PreAct_ResNet_Cifar(PreAct_bottleneck_block, [111, 111, 111], **kwargs)
    return model


