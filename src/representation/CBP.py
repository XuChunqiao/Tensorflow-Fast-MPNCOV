import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np

@tf.custom_gradient
def _signed_sqrt(input,thresh=1e-1):
    x = input
    y = tf.sign(x)*tf.sqrt(tf.abs(x))
    def grad(dy):
        grad_input = 0.5 * tf.math.rsqrt(tf.abs(x)+thresh)
        grad_input = grad_input * dy
        return grad_input, None
    return y, grad


class CompactBilinearLayer(layers.Layer):
    """Compact Bilinear Pooling
           implementation of Compact Bilinear Pooling (CBP)
           https://arxiv.org/pdf/1511.06062.pdf

        Args:
            thresh: small positive number for computation stability
            proj_dim: projected dimension
            input_dim: the #channel of input feature
        """
    def __init__(self, thresh=1e-8, proj_dim=8192, input_dim=256):
        super(CompactBilinearLayer, self).__init__()
        self.thresh = thresh
        self.output_dim = proj_dim
        self.input_dim = input_dim

        tf.random.set_seed(3)
        self.h_ = [
            tf.random.uniform(shape=(self.input_dim,), minval=0, maxval=self.output_dim, dtype=tf.int64),
            tf.random.uniform(shape=(self.input_dim,), minval=0, maxval=self.output_dim, dtype=tf.int64)
        ]
        tf.random.set_seed(5)
        self.weights_ = [
            tf.sign(tf.random.uniform(shape=(self.input_dim,), minval=-1, maxval=1, dtype=tf.float32)),
            tf.sign(tf.random.uniform(shape=(self.input_dim,), minval=-1, maxval=1, dtype=tf.float32))
        ]
        self.indices = [
            tf.concat([tf.reshape(tf.range(self.input_dim, dtype=tf.int64), [-1, 1]),
                       tf.reshape(self.h_[0], [-1, 1])], axis=1),
            tf.concat([tf.reshape(tf.range(self.input_dim, dtype=tf.int64), [-1, 1]),
                       tf.reshape(self.h_[1], [-1, 1])], axis=1)
        ]
        self.sparseM = [
            tf.sparse.to_dense(tf.sparse.SparseTensor(indices=self.indices[0], values=self.weights_[0],
                                                      dense_shape=[self.input_dim, self.output_dim])),
            tf.sparse.to_dense(tf.sparse.SparseTensor(indices=self.indices[1], values=self.weights_[1],
                                                      dense_shape=[self.input_dim, self.output_dim]))
        ]

    def _signed_sqrt(self, x):
        x = _signed_sqrt(x, self.thresh)
        return x

    def _l2norm(self, x):
        x = tf.nn.l2_normalize(x, axis=1, epsilon=1e-10)
        return x

    def call(self, x, training=None):
        bsn = 1
        batchSize, h, w, dim = x.shape
        x_flat = tf.reshape(x, [-1, dim])  # batchsize,h, w, dim,
        y = None

        for img in range(batchSize // bsn):
            segLen = bsn * h * w
            upper = batchSize * h * w
            x_start = img * segLen
            x_end = min(upper, (img + 1) * segLen)

            batch_x = x_flat[x_start:x_end]
            sketch1 = tf.signal.fft(tf.cast(tf.matmul(batch_x, self.sparseM[0]), dtype=tf.complex64))
            sketch2 = tf.signal.fft(tf.cast(tf.matmul(batch_x, self.sparseM[1]), dtype=tf.complex64))

            sketch = sketch1 * sketch2
            tmp_y = tf.cast(tf.signal.ifft(sketch), dtype=tf.float32)
            if y is None:
                y = tf.reshape(tmp_y, [bsn, h, w, self.output_dim])
            else:
                y = tf.concat([y, tf.reshape(tmp_y, [bsn, h, w, self.output_dim])], axis=0)
        y = tf.reduce_sum(y, axis=[1, 2])
        y = self._signed_sqrt(y)
        y = self._l2norm(y)
        return y
