"""CIFAR-10 data set.

See http://www.cs.toronto.edu/~kriz/cifar.html.
"""
import os

import tensorflow as tf

HEIGHT = 32
WIDTH = 32
DEPTH = 3


class CifarDataSet(object):
    """Cifar data set."""

    def __init__(self,
                 data_dir,
                 is_training=True,
                 pretrained=False,
                 arch=None,
                 imb_factor=None,
                 use_distortion=True):
        self.data_dir = data_dir
        self.is_training = is_training
        self.pretrained = pretrained
        self.arch = arch
        self.imb_factor = imb_factor
        self.use_distortion = use_distortion

    def get_filenames(self):

        if self.is_training:
            return [os.path.join(self.data_dir, 'train' + '.tfrecords')]
        else:
            return [os.path.join(self.data_dir, 'eval' + '.tfrecords')]

    def parser(self, serialized_example):
        """Parses a single tf.Example into image and label tensors."""
        # Dimensions of the images in the CIFAR-10 dataset.
        # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of
        # the input format.
        features = tf.io.parse_single_example(
            serialized_example,
            features={
                'image': tf.io.FixedLenFeature([], tf.string),
                'label': tf.io.FixedLenFeature([], tf.int64),
            })
        image = tf.io.decode_raw(features['image'], tf.uint8)
        image.set_shape([DEPTH * HEIGHT * WIDTH])

        # Reshape from [depth * height * width] to [depth, height, width].
        image = tf.cast(
            tf.transpose(tf.reshape(image, [DEPTH, HEIGHT, WIDTH]), [1, 2, 0]),
            tf.float32)
        label = tf.cast(features['label'], tf.int32)

        # Custom preprocessing.
        image = self.preprocess(image)

        return image, label

    def make_source_dataset(self, batchsize):
        """Read the images and labels from 'filenames'."""
        filenames = self.get_filenames()
        # Repeat infinitely.
        dataset = tf.data.TFRecordDataset(filenames)

        # Parse records.
        dataset = dataset.map(self.parser, num_parallel_calls=batchsize)

        # Potentially shuffle records.
        if self.is_training:
            min_queue_examples = int(
                CifarDataSet.num_examples_per_epoch(self.is_training) * 0.4)
            # Ensure that the capacity is sufficiently large to provide good
            # random shuffling.
            dataset = dataset.shuffle(buffer_size=min_queue_examples + 3 * batchsize)
        # Batch it up.
        dataset = dataset.batch(batchsize)
        return dataset

    def preprocess(self, image):
        """Preprocess a single image in [height, width, depth] layout."""
        if self.is_training and self.use_distortion:
            # Pad 4 pixels on each dimension of feature map, done in mini-batch
            image = tf.image.resize_with_crop_or_pad(image, 40, 40)
            image = tf.image.random_crop(image, [HEIGHT, WIDTH, DEPTH])
            image = tf.image.random_flip_left_right(image)
        mean = [0.507*255, 0.487*255, 0.441*255]
        std = [0.267*255, 0.256*255, 0.276*255]
        image = tf.subtract(image, mean)
        image = tf.divide(image, std)
        return image

    @staticmethod
    def num_examples_per_epoch(is_training):
        if is_training:
            return 50000
        else:
            return 10000
