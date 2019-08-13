from src.network import *
import tensorflow as tf
import numpy as np
import warnings
__all__ = ['Newmodel', 'get_model']

class Newmodel(Basemodel):
    """replace the image representation method and classifier

       Args:
       modeltype: model archtecture
       representation: image representation method
       num_classes: the number of classes
       freezed_layer: the end of freezed layers in network
       pretrained: whether use pretrained weights or not
    """
    def __init__(self, modeltype, representation, num_classes, freezed_layer, pretrained=False):
        super(Newmodel, self).__init__(modeltype, pretrained)
        if representation is not None:
            representation_method = representation['function']
            representation.pop('function')
            representation_args = representation
            representation_args['input_dim'] = self.representation_dim
            self.representation = representation_method(**representation_args)

            if not self.pretrained:
                if modeltype.startswith('vgg'):
                    self.classifier.pop()
                    self.classifier.add(tf.keras.layers.Dense(num_classes))
                else:
                    self.classifier = tf.keras.layers.Dense(num_classes)
            else:
                self.classifier = tf.keras.layers.Dense(num_classes, name='classifier')

        else:
            if modeltype.startswith('vgg'):
                self.classifier.pop()
                self.classifier.add(tf.keras.layers.Dense(num_classes))
            else:
                self.classifier = tf.keras.layers.Dense(num_classes, name='classifier')


        if freezed_layer:
            model_layers = self.features.layers + self.representation.layers + self.classifier.layers
            for i in range(freezed_layer):
                model_layers[i].trainable=False



def get_model(modeltype, representation, num_classes, freezed_layer, pretrained=False):
    _model = Newmodel(modeltype, representation, num_classes, freezed_layer, pretrained=pretrained)
    input = tf.random.normal([1,224,224,3])
    # _model.features.trainable = False
    _model(input, training=False)
    _model.features.summary()
    # _model.representation
    # _model.classifier.summary()
    _model.summary()

    return _model

