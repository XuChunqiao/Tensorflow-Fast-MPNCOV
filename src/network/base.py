from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import warnings

from .mpncov_resnet import *
from .resnet import *
from .vgg import *


def get_basemodel(modeltype, pretrained=False):
    modeltype = globals()[modeltype]
    if pretrained == False:
        warnings.warn('You will use model that randomly initialized!')
    return modeltype(pretrained=pretrained)

class Basemodel(tf.keras.Model):
    """Load backbone model and reconstruct it into three part:
       1) feature extractor
       2) global image representaion
       3) classifier
    """
    def __init__(self, modeltype, pretrained=False):
        super(Basemodel, self).__init__()
        basemodel = get_basemodel(modeltype, pretrained)
        self.pretrained = pretrained
        if modeltype.startswith('resnet'):
            basemodel = self._reconstruct_resnet(basemodel)
        elif modeltype.startswith('mpncovresnet'):
            basemodel = self._reconstruct_mpncovresnet(basemodel)
        elif modeltype.startswith('vgg'):
            basemodel = self._reconstruct_vgg(basemodel)

        self.features = basemodel.features
        self.representation = basemodel.representation
        self.classifier = basemodel.classifier
        self.representation_dim = basemodel.representation_dim

    def _reconstruct_resnet(self, basemodel):
        model = tf.keras.Model()
        model.features = tf.keras.Sequential(layers=basemodel.layers[:-2], name='features')
        model.representation = basemodel.avgpool
        model.classifier = basemodel.fc
        model.representation_dim = 2048
        return model

    def _reconstruct_mpncovresnet(self, basemodel):
        model = tf.keras.Model()
        model.features = tf.keras.Sequential(layers=basemodel.layers[:-2], name='features')
        model.representation = basemodel.MPNCOV
        model.classifier = basemodel.fc
        model.representation_dim = 256
        return model

    def _reconstruct_vgg(self, basemodel):
        model = tf.keras.Model()
        model.features = tf.keras.Sequential(layers=basemodel.features.layers[:-1], name='features')
        model.representation = basemodel.features.layers[-1]
        model.classifier = basemodel.classifier
        model.representation_dim = 512
        return model

    def call(self, x, training=None):
        x = self.features(x, training=training)
        x = self.representation(x, training=training)
        out = self.classifier(x)
        return out
