'''
@file: MPNCOV.py
@author: Chunqiao Xu
@author: Jiangtao Xie
@author: Peihua Li
Please cite the paper below if you use the code:

Peihua Li, Jiangtao Xie, Qilong Wang and Zilin Gao. Towards Faster Training of Global Covariance Pooling Networks by Iterative Matrix Square Root Normalization. IEEE Int. Conf. on Computer Vision and Pattern Recognition (CVPR), pp. 947-955, 2018.

Peihua Li, Jiangtao Xie, Qilong Wang and Wangmeng Zuo. Is Second-order Information Helpful for Large-scale Visual Recognition? IEEE Int. Conf. on Computer Vision (ICCV),  pp. 2070-2078, 2017.

Copyright (C) 2018 Peihua Li and Jiangtao Xie

All rights reserved.
'''
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow.keras.layers as layers


class MPNCOV(tf.keras.Model):
    """Matrix power normalized Covariance pooling (MPNCOV)
           implementation of fast MPN-COV (i.e.,iSQRT-COV)
           https://arxiv.org/abs/1712.01034

        Args:
            iterNum: #iteration of Newton-schulz method
            input_dim: the #channel of input feature
            dimension_reduction: if None, it will not use 1x1 conv to
                                  reduce the #channel of feature.
                                 if 256 or others, the #channel of feature
                                  will be reduced to 256 or others.
        """

    def __init__(self, iterNum=5, input_dim=2048, dimension_reduction=None, dropout_p=None):
        super(MPNCOV, self).__init__()
        self.iterNum = iterNum
        self.dr = dimension_reduction
        self.dropout_p = dropout_p
        self.input_dim = input_dim

        if self.dr is not None:
            if tf.keras.backend.image_data_format() == 'channels_last':
                axis = 3
            else:
                axis = 1
            self.conv_dr_block = tf.keras.Sequential(
                layers=[layers.Conv2D(self.dr, kernel_size=1, strides=1, use_bias=False, padding='same',
                                      kernel_initializer=tf.keras.initializers.VarianceScaling()),
                        layers.BatchNormalization(axis=axis,
                                                  momentum=0.9,
                                                  epsilon=1e-5,
                                                  center=True,
                                                  scale=True,
                                                  fused=True,
                                                  gamma_initializer=tf.ones_initializer()),
                        layers.ReLU()],
                name='conv_dr_block')
            output_dim = self.dr
        else:
            output_dim = input_dim
        self.output_dim = int(output_dim * (output_dim + 1) / 2)

        if self.dropout_p is not None:
            self.dropout = tf.keras.layers.Dropout(self.dropout_p)

    def call(self, x, training=None):
        if self.dr is not None:
            x = self.conv_dr_block(x, training=training)

        x = Covpool(x)
        x = Sqrtm(x)
        x = Triuvec(x)


        if self.dropout_p is not None:
            x = self.dropout(x, training=training)
        return x

@tf.custom_gradient
def Covpool(input):
    x = input
    batchSize, h, w, dim = x.shape
    M = int(h * w)
    dtype = x.dtype
    x = tf.reshape(x, [batchSize, M, dim])
    I_hat = tf.cast(tf.divide(-1., M*M) * tf.ones([M, M]) + tf.divide(1., M) * tf.eye(M), dtype)
    I_hat = tf.expand_dims(I_hat, 0)
    I_hat = tf.tile(I_hat, [batchSize, 1, 1])
    y = tf.matmul(tf.matmul(x, I_hat, transpose_a=True), x)
    def grad(dy):
        grad_input = dy + tf.transpose(dy, [0, 2, 1])
        grad_input = tf.matmul(I_hat, tf.matmul(x, grad_input))
        grad_input = tf.reshape(grad_input, [batchSize, h, w, dim])
        return grad_input
    return y, grad


@tf.custom_gradient
def Sqrtm(input, iterN = 5):
    x = input
    # iterN = 5
    dtype = x.dtype
    batchSize = x.shape[0]
    dim = x.shape[1]
    I3 = tf.cast(3.0 * tf.tile(tf.expand_dims(tf.eye(dim), 0), [batchSize,1,1]), dtype)
    normA = tf.reduce_sum(tf.reduce_sum((1./3.) * x * I3, axis=1, keepdims=True), axis=2, keepdims=True)
    A = tf.divide(x, tf.tile(normA,[1, dim, dim]))
    Y = [tf.cast(tf.zeros([batchSize, dim, dim]), dtype) for i in range(iterN)]
    Z = [tf.cast(tf.tile(tf.expand_dims(tf.eye(dim), 0), [batchSize, 1, 1]), dtype) for i in range(iterN)]

    if iterN < 2:
        ZY = 0.5*(I3 - A)
        YZY = tf.matmul(ZY, A)
    else:
        ZY = 0.5*(I3 - A)
        Y[0] = tf.matmul(ZY, A)
        Z[0] = ZY
        for i in range(1, iterN):
           ZY = 0.5 * (I3 - tf.matmul( Z[i-1], Y[i-1]))
           Y[i] = tf.matmul(Y[i-1], ZY)
           Z[i] = tf.matmul(ZY, Z[i-1])
        YZY = 0.5 * tf.matmul(Y[iterN-2], I3 - tf.matmul(Z[iterN-2], Y[iterN-2]))
    y = YZY * tf.tile(tf.sqrt(normA), [1, dim, dim])
    def grad(dy):
        der_postCom = dy * tf.tile(tf.sqrt(normA), [1, dim, dim])
        der_postComAux = tf.divide(tf.reduce_sum(tf.reduce_sum(dy * YZY, axis=1, keepdims=True), axis=2, keepdims=True),2 * tf.sqrt(normA))
        if iterN < 2:
            der_NSiter = 0.5 * tf.subtract(tf.matmul(der_postCom, I3 - A), tf.matmul(A, der_postCom))
        else:
            dldY = 0.5 * tf.subtract(tf.matmul(der_postCom, I3 - tf.matmul(Y[iterN-2], Z[iterN - 2])),
                                     tf.matmul(tf.matmul(Z[iterN - 2], Y[iterN - 2]), der_postCom))
            dldZ = -0.5 * tf.matmul(tf.matmul(Y[iterN - 2], der_postCom), Y[iterN - 2])
            for i in range(iterN - 3, -1, -1):
                YZ = I3 - tf.matmul(Y[i], Z[i])
                ZY = tf.matmul(Z[i], Y[i])
                dldY_ = 0.5 * tf.subtract(tf.subtract(tf.matmul(dldY, YZ),
                                                      tf.matmul(tf.matmul(Z[i], dldZ), Z[i])),
                                          tf.matmul(ZY, dldY))
                dldZ_ = 0.5 * tf.subtract(tf.subtract(tf.matmul(YZ, dldZ),
                                                      tf.matmul(tf.matmul(Y[i], dldY), Y[i])),
                                          tf.matmul(dldZ, ZY))
                dldY = dldY_
                dldZ = dldZ_
            der_NSiter = 0.5 * tf.subtract(tf.subtract(tf.matmul(dldY, I3 - A), dldZ),
                                           tf.matmul(A, dldY))
        # der_NSiter = der_NSiter.transpose(1, 2)
        grad_input = tf.divide(der_NSiter, tf.tile(normA, [1, dim, dim]))
        grad_aux = tf.reduce_sum(tf.reduce_sum(der_NSiter * x, axis=1, keepdims=True), axis=2, keepdims=True)
        grad_input += tf.tile(tf.subtract(der_postComAux, tf.divide(grad_aux, (normA * normA))), [1, dim, dim]) \
                      * tf.cast(tf.tile(tf.expand_dims(tf.eye(dim), 0), [batchSize, 1, 1]), dtype)
        return grad_input
    return y, grad


@tf.custom_gradient
def Triuvec(input):
    x = input
    batchSize = x.shape[0]
    dim = x.shape[1]
    x = tf.reshape(x, [batchSize, dim*dim])
    index = tf.where(tf.tile(tf.expand_dims(tf.reshape(tf.linalg.band_part(tf.ones([dim, dim]), 0, -1), [dim*dim]), 0), [batchSize, 1]) > 0)
    y = tf.gather_nd(x, tf.reshape(index, [batchSize, int(dim * (dim + 1) / 2), 2]))
    def grad(dy):
        index = tf.where(tf.tile(tf.expand_dims(tf.reshape(tf.linalg.band_part(tf.ones([dim, dim]), 0, -1), [dim * dim]), 0),[batchSize, 1]) > 0)
        index = tf.reshape(index, [batchSize, int(dim * (dim + 1) / 2), 2])
        grad_input = tf.scatter_nd(index, dy, [batchSize, dim*dim])
        grad_input = tf.reshape(grad_input, [batchSize, dim, dim])
        return grad_input
    return y, grad




