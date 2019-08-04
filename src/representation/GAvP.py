import tensorflow as tf

class GAvP(tf.keras.Model):
     """Global Average pooling
        Widely used in ResNet, Inception, DenseNet, etc.
        feature_crop: tuple(kernel_size, step)
                        if None, it will not execute step-crop operation at validation
     """
     def __init__(self, input_dim=2048):
         super(GAvP, self).__init__()
         self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
         self.output_dim = input_dim

     def call(self, x, training=None):
         x = self.avgpool(x)
         return x