"""
@file BCNN.py
@author: Chunqiao Xu
@author: JiangtaoXie
@author: Peihua Li
"""

import tensorflow as tf
class BCNN(tf.keras.Model):
    """
        BCNN: implementation of Bilinear CNN
              https://arxiv.org/pdf/1504.07889.pdf
        Args:
            input_dim: the #channel of input feature
            thresh: small positive number for computation stability
    """
    def __init__(self, input_dim=512, thresh=1e-8):
        super(BCNN, self).__init__()
        self.input_dim = input_dim
        self.thresh = thresh
        self.output_dim = self.input_dim**2

    def call(self, x, training=None):
        x = _bilinearpool(x)
        x = _signed_sqrt(x)
        x = tf.reshape(x, [x.shape[0], -1])
        x = _l2norm(x)
        return x
@tf.custom_gradient
def _signed_sqrt(input, thresh=1e-8):
    x = input
    x = tf.sign(x)*tf.sqrt(tf.abs(x))
    def grad(dy):
        gradinput = 0.5*tf.math.rsqrt(tf.abs(x)+thresh)
        gradinput = gradinput*dy
        return gradinput, None
    return x, grad

def _bilinearpool(input):
    x = input
    batchSize, h, w, dim = x.shape
    x = tf.reshape(x, [batchSize, h*w, dim])
    x = 1/(h*w)*tf.matmul(tf.transpose(x, perm=[0,2,1]), x)
    return x
def _l2norm(input):
    x = input
    x = tf.nn.l2_normalize(x, axis=1, epsilon=1e-10)
    return x