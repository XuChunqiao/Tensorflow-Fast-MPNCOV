from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
from PIL import Image

import os
import os.path
import scipy.io as sio

AUTOTUNE = tf.data.experimental.AUTOTUNE

class CUB_dataset(object):
    def __init__(self, is_training=True, data_dir=None, pretrained=False, arch=None,):
        """Create  TFRecord files from Raw Images and Create an input from TFRecord files.
        Args:
          is_training: `bool` for whether the input is for training
          data_dir: `str` for the directory of the training and validation data;
              if 'null' (the literal string 'null') or implicitly False
              then construct a null pipeline, consisting of empty images
              and blank labels.
        """
        super(CUB_dataset, self).__init__()
        IMAGESIZE = 448
        self.is_training = is_training
        self.data_dir = data_dir
        self.pretrained = pretrained
        self.arch = arch

        def preprocess_train_image_randomflip(image_bytes):
            shape = tf.image.extract_jpeg_shape(image_bytes)
            image_height = shape[0]
            image_width = shape[1]

            padded_center_crop_size = tf.cast(tf.minimum(image_height, image_width), tf.int32)

            offset_height = ((image_height - padded_center_crop_size) + 1) // 2
            offset_width = ((image_width - padded_center_crop_size) + 1) // 2
            crop_window = tf.stack([offset_height, offset_width,
                                    padded_center_crop_size, padded_center_crop_size])
            image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
            image = tf.image.resize(image, [IMAGESIZE, IMAGESIZE])
            image = tf.image.random_flip_left_right(image)
            if self.pretrained and self.arch.startswith('vgg'):
                # RGB==>BGR for VGG16
                image = image[..., ::-1]
                mean = [0.406 * 255, 0.456 * 255, 0.485 * 255]
                std = None
            else:
                mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
                std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
            if mean is not None:
                image = tf.subtract(image, mean)
            if std is not None:
                image = tf.divide(image, std)

            return image

        def preprocess_val_image_flip(image_bytes):
            shape = tf.image.extract_jpeg_shape(image_bytes)
            image_height = shape[0]
            image_width = shape[1]

            padded_center_crop_size = tf.cast(tf.minimum(image_height, image_width), tf.int32)

            offset_height = ((image_height - padded_center_crop_size) + 1) // 2
            offset_width = ((image_width - padded_center_crop_size) + 1) // 2
            crop_window = tf.stack([offset_height, offset_width,
                                    padded_center_crop_size, padded_center_crop_size])
            image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
            image = tf.image.resize(image, [IMAGESIZE, IMAGESIZE])
            if self.pretrained and self.arch.startswith('vgg'):
                # RGB==>BGR for VGG16
                image = image[..., ::-1]
                mean = [0.406 * 255, 0.456 * 255, 0.485 * 255]
                std = None
            else:
                mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
                std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

            image_flip = tf.image.flip_left_right(image)
            image = tf.stack([image, image_flip])
            if mean is not None:
                image = tf.subtract(image, mean)
            if std is not None:
                image = tf.divide(image, std)
            return image

        if self.is_training:
            dataset_dir = os.path.join(self.data_dir, 'train')
            tfrecord_filename = os.path.join(self.data_dir, 'train.tfrec')
            label_filename = os.path.join(self.data_dir, 'train_label.mat')
            preprocess_image = preprocess_train_image_randomflip

        else:
            dataset_dir = os.path.join(self.data_dir, 'val')
            tfrecord_filename = os.path.join(self.data_dir, 'val.tfrec')
            label_filename = os.path.join(self.data_dir, 'val_label.mat')
            preprocess_image = preprocess_val_image_flip

        # generate tfrecord files
        if not os.path.exists(tfrecord_filename):
            species = sorted(os.listdir(dataset_dir))
            img_path_id = []
            for i in range(len(species)):
                s_dir = os.path.join(dataset_dir, species[i])
                # 遍历目录下的所有图片
                for filename in os.listdir(s_dir):
                    # 获取文件的路径
                    file_path = os.path.join(s_dir, filename)
                    if file_path.endswith("jpg") and os.path.exists(file_path):
                        img_path_id.append([file_path, i])
            img_path_id = np.asarray(img_path_id)
            np.random.shuffle(img_path_id)
            images = img_path_id[:, 0].tolist()
            labels = np.int64(img_path_id[:, 1])
            sio.savemat(label_filename, {'label': labels})

            tfrec = tf.data.experimental.TFRecordWriter(tfrecord_filename)
            image_ds = tf.data.Dataset.from_tensor_slices(images).map(tf.io.read_file)
            tfrec.write(image_ds)

        labels = sio.loadmat(label_filename)['label'][0]
        image_ds = tf.data.TFRecordDataset(tfrecord_filename).map(preprocess_image)
        label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labels, tf.int64))

        self.ds = tf.data.Dataset.zip((image_ds, label_ds))
        self.num_samples = len(labels)

    def make_source_dataset(self, batchsize):
        if self.is_training:
            self.ds = self.ds.shuffle(buffer_size=int(0.4*self.num_samples))
        return self.ds.batch(batchsize).prefetch(buffer_size=AUTOTUNE)


class Aircrafts_dataset(object):
    def __init__(self, is_training=True, data_dir=None, pretrained=False, arch=None):
        """Create  TFRecord files from Raw Images and Create an input from TFRecord files.
        Args:
          is_training: `bool` for whether the input is for training
          data_dir: `str` for the directory of the training and validation data;
              if 'null' (the literal string 'null') or implicitly False
              then construct a null pipeline, consisting of empty images
              and blank labels.
        """
        super(Aircrafts_dataset, self).__init__()
        IMAGESIZE = 512
        self.is_training = is_training
        self.data_dir = data_dir
        self.pretrained = pretrained
        self.arch = arch

        def preprocess_train_image_randomflip(image_bytes):
            image = tf.image.decode_jpeg(image_bytes, channels=3)
            image = tf.image.resize(image, [IMAGESIZE, IMAGESIZE])
            image = tf.image.central_crop(image, 448.0 / IMAGESIZE)
            image = tf.image.random_flip_left_right(image)
            if self.pretrained and self.arch.startswith('vgg'):
                # RGB==>BGR for VGG16
                image = image[..., ::-1]
                mean = [0.406 * 255, 0.456 * 255, 0.485 * 255]
                std = None
            else:
                mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
                std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
            if mean is not None:
                image = tf.subtract(image, mean)
            if std is not None:
                image = tf.divide(image, std)
            return image

        def preprocess_val_image_flip(image_bytes):
            image = tf.image.decode_jpeg(image_bytes, channels=3)
            image = tf.image.resize(image, [IMAGESIZE, IMAGESIZE])
            image = tf.image.central_crop(image, 448.0 / IMAGESIZE)
            if self.pretrained and self.arch.startswith('vgg'):
                # RGB==>BGR for VGG16
                image = image[..., ::-1]
                mean = [0.406 * 255, 0.456 * 255, 0.485 * 255]
                std = None
            else:
                mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
                std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

            image_flip = tf.image.flip_left_right(image)
            image = tf.stack([image, image_flip])
            if mean is not None:
                image = tf.subtract(image, mean)
            if std is not None:
                image = tf.divide(image, std)
            return image

        if self.is_training:
            dataset_dir = os.path.join(self.data_dir, 'train')
            tfrecord_filename = os.path.join(self.data_dir, 'train.tfrec')
            label_filename = os.path.join(self.data_dir, 'train_label.mat')
            preprocess_image = preprocess_train_image_randomflip

        else:
            dataset_dir = os.path.join(self.data_dir, 'val')
            tfrecord_filename = os.path.join(self.data_dir, 'val.tfrec')
            label_filename = os.path.join(self.data_dir, 'val_label.mat')
            preprocess_image = preprocess_val_image_flip

        if not os.path.exists(tfrecord_filename):
            species = sorted(os.listdir(dataset_dir))
            img_path_id = []
            for i in range(len(species)):
                s_dir = os.path.join(dataset_dir, species[i])
                # 遍历目录下的所有图片
                for filename in os.listdir(s_dir):
                    # 获取文件的路径
                    file_path = os.path.join(s_dir, filename)
                    if file_path.endswith("jpg") and os.path.exists(file_path):
                        img_path_id.append([file_path, i])
            img_path_id = np.asarray(img_path_id)
            np.random.shuffle(img_path_id)
            images = img_path_id[:, 0].tolist()
            labels = np.int64(img_path_id[:, 1])
            sio.savemat(label_filename, {'label': labels})

            tfrec = tf.data.experimental.TFRecordWriter(tfrecord_filename)
            image_ds = tf.data.Dataset.from_tensor_slices(images).map(tf.io.read_file)
            tfrec.write(image_ds)

        labels = sio.loadmat(label_filename)['label'][0]
        image_ds = tf.data.TFRecordDataset(tfrecord_filename).map(preprocess_image)
        label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labels, tf.int64))
        self.ds = tf.data.Dataset.zip((image_ds, label_ds))
        self.num_samples = len(labels)

    def make_source_dataset(self, batchsize):
        if self.is_training:
            self.ds = self.ds.shuffle(buffer_size=int(0.4*self.num_samples))
        return self.ds.batch(batchsize).prefetch(buffer_size=AUTOTUNE)

class Cars_dataset(object):
    def __init__(self, is_training=True, data_dir=None, pretrained=False, arch=None):
        """Create  TFRecord files from Raw Images and Create an input from TFRecord files.
        Args:
          is_training: `bool` for whether the input is for training
          data_dir: `str` for the directory of the training and validation data;
              if 'null' (the literal string 'null') or implicitly False
              then construct a null pipeline, consisting of empty images
              and blank labels.
        """
        super(Cars_dataset, self).__init__()
        IMAGESIZE = 448
        self.is_training = is_training
        self.data_dir = data_dir
        self.pretrained = pretrained
        self.arch = arch

        def preprocess_train_image_randomflip(image_bytes):
            image = tf.image.decode_jpeg(image_bytes, channels=3)
            image = tf.image.resize(image, [IMAGESIZE, IMAGESIZE])
            image = tf.image.random_flip_left_right(image)
            if self.pretrained and self.arch.startswith('vgg'):
                # RGB==>BGR for VGG16
                image = image[..., ::-1]
                mean = [0.406 * 255, 0.456 * 255, 0.485 * 255]
                std = None
            else:
                mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
                std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
            if mean is not None:
                image = tf.subtract(image, mean)
            if std is not None:
                image = tf.divide(image, std)
            return image

        def preprocess_val_image_flip(image_bytes):
            image = tf.image.decode_jpeg(image_bytes, channels=3)
            image = tf.image.resize(image, [IMAGESIZE, IMAGESIZE])
            if self.pretrained and self.arch.startswith('vgg'):
                # RGB==>BGR for VGG16
                image = image[..., ::-1]
                mean = [0.406 * 255, 0.456 * 255, 0.485 * 255]
                std = None
            else:
                mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
                std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

            image_flip = tf.image.flip_left_right(image)
            image = tf.stack([image, image_flip])
            if mean is not None:
                image = tf.subtract(image, mean)
            if std is not None:
                image = tf.divide(image, std)
            return image

        if self.is_training:
            dataset_dir = os.path.join(self.data_dir, 'train')
            tfrecord_filename = os.path.join(self.data_dir, 'train.tfrec')
            label_filename = os.path.join(self.data_dir, 'train_label.mat')
            preprocess_image = preprocess_train_image_randomflip
        else:
            dataset_dir = os.path.join(self.data_dir, 'val')
            tfrecord_filename = os.path.join(self.data_dir, 'val.tfrec')
            label_filename = os.path.join(self.data_dir, 'val_label.mat')
            preprocess_image = preprocess_val_image_flip

        if not os.path.exists(tfrecord_filename):
            species = sorted(os.listdir(dataset_dir))
            img_path_id = []
            for i in range(len(species)):
                s_dir = os.path.join(dataset_dir, species[i])
                # 遍历目录下的所有图片
                for filename in os.listdir(s_dir):
                    # 获取文件的路径
                    file_path = os.path.join(s_dir, filename)
                    if file_path.endswith("jpg") and os.path.exists(file_path):
                        img_path_id.append([file_path, i])
            img_path_id = np.asarray(img_path_id)
            np.random.shuffle(img_path_id)
            images = img_path_id[:, 0].tolist()
            labels = np.int64(img_path_id[:, 1])
            sio.savemat(label_filename, {'label': labels})

            tfrec = tf.data.experimental.TFRecordWriter(tfrecord_filename)
            image_ds = tf.data.Dataset.from_tensor_slices(images).map(tf.io.read_file)
            tfrec.write(image_ds)

        labels = sio.loadmat(label_filename)['label'][0]
        image_ds = tf.data.TFRecordDataset(tfrecord_filename).map(preprocess_image)
        label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labels, tf.int64))
        self.ds = tf.data.Dataset.zip((image_ds, label_ds))
        self.num_samples = len(labels)

    def make_source_dataset(self, batchsize):
        if self.is_training:
            self.ds = self.ds.shuffle(buffer_size=int(0.4*self.num_samples))
        return self.ds.batch(batchsize).prefetch(buffer_size=AUTOTUNE)
