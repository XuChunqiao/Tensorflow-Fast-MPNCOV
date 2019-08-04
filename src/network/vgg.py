from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow.keras.layers as layers

__all__ = ['VGG','vgg16', 'vgg19']
class VGG(tf.keras.Model):
    def __init__(self, features, classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = tf.keras.Sequential(
            layers=[
                layers.Flatten(),
                layers.Dense(4096, activation='relu'),
                layers.Dense(4096, activation='relu'),
                layers.Dense(classes)],
            name='classifier')
    def call(self, x, training=None):
        x = self.features(x, training=training)
        x = self.classifier(x)
        return x


def vgg16(pretrained):
    features = tf.keras.Sequential(
        layers=[
            # Block 1
            layers.Conv2D(64, (3, 3), activation='relu', padding="same", name='block1_conv1'),
            layers.Conv2D(64, (3, 3), activation='relu', padding="same", name='block1_conv2'),
            layers.MaxPooling2D((2, 2), padding="valid", name='block1_pool'),
            # Block 2
            layers.Conv2D(128, (3, 3), activation='relu', padding="same", name='block2_conv1'),
            layers.Conv2D(128, (3, 3), activation='relu', padding="same", name='block2_conv2'),
            layers.MaxPooling2D((2, 2), padding="valid", name='block2_pool'),
            # Block 3
            layers.Conv2D(256, (3, 3), activation='relu', padding="same", name='block3_conv1'),
            layers.Conv2D(256, (3, 3), activation='relu', padding="same", name='block3_conv2'),
            layers.Conv2D(256, (3, 3), activation='relu', padding="same", name='block3_conv3'),
            layers.MaxPooling2D((2, 2), padding="valid", name='block3_pool'),
            # Block 4
            layers.Conv2D(512, (3, 3), activation='relu', padding="same", name='block4_conv1'),
            layers.Conv2D(512, (3, 3), activation='relu', padding="same", name='block4_conv2'),
            layers.Conv2D(512, (3, 3), activation='relu', padding="same", name='block4_conv3'),
            layers.MaxPooling2D((2, 2), padding="valid", name='block4_pool'),
            # Block 5
            layers.Conv2D(512, (3, 3), activation='relu', padding="same", name='block5_conv1'),
            layers.Conv2D(512, (3, 3), activation='relu', padding="same", name='block5_conv2'),
            layers.Conv2D(512, (3, 3), activation='relu', padding="same", name='block5_conv3'),
            layers.MaxPooling2D((2, 2), padding="valid", name='block5_pool')],
        name='vgg16_features')
    model = VGG(features)
    if pretrained:
        input = tf.random.normal([1, 224, 224, 3])
        model(input)
        model.features.summary()
        model.classifier.summary()
        base_model = tf.keras.applications.VGG16(include_top=True,
                                                 weights='imagenet')  # 加载预训练模型
        weights = base_model.get_weights()
        model.set_weights(weights)
        del base_model, weights
    return model

def vgg19(pretrained):
    features = tf.keras.Sequential(
        layers=[
            # Block 1
            layers.Conv2D(64, (3, 3), activation='relu', padding="same", name='block1_conv1'),
            layers.Conv2D(64, (3, 3), activation='relu', padding="same", name='block1_conv2'),
            layers.MaxPooling2D((2, 2), padding="valid", name='block1_pool'),
            # Block 2
            layers.Conv2D(128, (3, 3), activation='relu', padding="same", name='block2_conv1'),
            layers.Conv2D(128, (3, 3), activation='relu', padding="same", name='block2_conv2'),
            layers.MaxPooling2D((2, 2), padding="valid", name='block2_pool'),
            # Block 3
            layers.Conv2D(256, (3, 3), activation='relu', padding="same", name='block3_conv1'),
            layers.Conv2D(256, (3, 3), activation='relu', padding="same", name='block3_conv2'),
            layers.Conv2D(256, (3, 3), activation='relu', padding="same", name='block3_conv3'),
            layers.Conv2D(256, (3, 3), activation='relu', padding="same", name='block3_conv4'),
            layers.MaxPooling2D((2, 2), padding="valid", name='block3_pool'),
            # Block 4
            layers.Conv2D(512, (3, 3), activation='relu', padding="same", name='block4_conv1'),
            layers.Conv2D(512, (3, 3), activation='relu', padding="same", name='block4_conv2'),
            layers.Conv2D(512, (3, 3), activation='relu', padding="same", name='block4_conv3'),
            layers.Conv2D(512, (3, 3), activation='relu', padding="same", name='block4_conv4'),
            layers.MaxPooling2D((2, 2), padding="valid", name='block4_pool'),
            # Block 5
            layers.Conv2D(512, (3, 3), activation='relu', padding="same", name='block5_conv1'),
            layers.Conv2D(512, (3, 3), activation='relu', padding="same", name='block5_conv2'),
            layers.Conv2D(512, (3, 3), activation='relu', padding="same", name='block5_conv3'),
            layers.Conv2D(512, (3, 3), activation='relu', padding="same", name='block5_conv4'),
            layers.MaxPooling2D((2, 2), padding="valid", name='block5_pool')],
        name='vgg19_features')
    model = VGG(features)

    if pretrained:
        input = tf.random.normal([1, 224, 224, 3])
        model(input)
        model.features.summary()
        model.classifier.summary()
        base_model = tf.keras.applications.VGG19(include_top=True,
                                                 weights='imagenet')  # 加载预训练模型
        weights = base_model.get_weights()
        model.set_weights(weights)
        del base_model, weights
    return model


