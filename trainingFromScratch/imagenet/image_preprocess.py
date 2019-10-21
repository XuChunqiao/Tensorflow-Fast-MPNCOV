# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""ImageNet preprocessing for ResNet."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import dtypes
import numpy as np

IMAGE_SIZE = 224
CROP_PADDING = 32
# IMAGE_SIZE = 320
# CROP_PADDING = 46
# IMAGE_SIZE = 448
# CROP_PADDING = 64


def distorted_bounding_box_crop(image_bytes,
                                bbox,
                                min_object_covered=0.1,
                                aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.05, 1.0),
                                max_attempts=100,
                                scope=None):
  """Generates cropped_image using one of the bboxes randomly distorted.

  See `tf.image.sample_distorted_bounding_box` for more documentation.

  Args:
    image_bytes: `Tensor` of binary image data.
    bbox: `Tensor` of bounding boxes arranged `[1, num_boxes, coords]`
        where each coordinate is [0, 1) and the coordinates are arranged
        as `[ymin, xmin, ymax, xmax]`. If num_boxes is 0 then use the whole
        image.
    min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
        area of the image must contain at least this fraction of any bounding
        box supplied.
    aspect_ratio_range: An optional list of `float`s. The cropped area of the
        image must have an aspect ratio = width / height within this range.
    area_range: An optional list of `float`s. The cropped area of the image
        must contain a fraction of the supplied image within in this range.
    max_attempts: An optional `int`. Number of attempts at generating a cropped
        region of the image of the specified constraints. After `max_attempts`
        failures, return the entire image.
    scope: Optional `str` for name scope.
  Returns:
    cropped image `Tensor`
  """
  with tf.name_scope('distorted_bounding_box_crop'):
    shape = tf.image.extract_jpeg_shape(image_bytes)
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        shape,
        bounding_boxes=bbox,
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, _ = sample_distorted_bounding_box

    # Crop the image to the specified bounding box.
    offset_y, offset_x, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)
    crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
    image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)

    return image


def _at_least_x_are_equal(a, b, x):
  """At least `x` of `a` and `b` `Tensors` are equal."""
  match = tf.equal(a, b)
  match = tf.cast(match, tf.int32)
  return tf.greater_equal(tf.reduce_sum(match), x)


def _decode_and_random_crop(image_bytes, image_size):
  """Make a random crop of image_size."""
  bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
  image = distorted_bounding_box_crop(
      image_bytes,
      bbox,
      min_object_covered=0.1,
      aspect_ratio_range=(3. / 4, 4. / 3.),
      area_range=(0.08, 1.0),
      max_attempts=10,
      scope=None)
  original_shape = tf.image.extract_jpeg_shape(image_bytes)
  bad = _at_least_x_are_equal(original_shape, tf.shape(image), 3)



  image = tf.cond(
      bad,
      lambda: _decode_and_center_crop(image_bytes, image_size),
      lambda: tf.image.resize(image,    # pylint: disable=g-long-lambda
                              [image_size, image_size],
                              method='bicubic'))


  return image


def _decode_and_center_crop(image_bytes, image_size):
  """Crops to center of image with padding then scales image_size."""
  shape = tf.image.extract_jpeg_shape(image_bytes)
  image_height = shape[0]
  image_width = shape[1]

  padded_center_crop_size = tf.cast(
      ((image_size / (image_size + CROP_PADDING)) *
       tf.cast(tf.minimum(image_height, image_width), tf.float32)),
      tf.int32)

  offset_height = ((image_height - padded_center_crop_size) + 1) // 2
  offset_width = ((image_width - padded_center_crop_size) + 1) // 2
  crop_window = tf.stack([offset_height, offset_width,
                          padded_center_crop_size, padded_center_crop_size])
  image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
  image = tf.image.resize(image, [image_size, image_size], method='bicubic')

  return image


def _flip(image):
  """Random horizontal image flip."""
  image = tf.image.random_flip_left_right(image)
  return image

def ContrastJitterAug(image, contrast):
    orgdtype = image.dtype
    if orgdtype in [dtypes.float16, dtypes.float32]:
        image = image
    else:
        image = tf.image.convert_image_dtype(image, dtypes.float32)

    coef = tf.convert_to_tensor([[[0.299,0.587,0.114]]])
    alpha = 1.0 + tf.random.uniform((1,),-contrast, contrast)
    gray = image * coef

    gray = 3.0 *(1.0-alpha)/tf.cast(tf.reduce_prod(tf.shape(gray)),dtypes.float32) * tf.reduce_sum(gray)
    image *= alpha
    image += gray
    return tf.image.convert_image_dtype(image, orgdtype, saturate=True)

def BrightnessJitterAug(image, brightness):
    orgdtype = image.dtype
    if orgdtype in [dtypes.float16, dtypes.float32]:
        image = image
    else:
        image = tf.image.convert_image_dtype(image, dtypes.float32)
    alpha = 1.0 + tf.random.uniform((1,), -brightness, brightness)
    image *= alpha

    return tf.image.convert_image_dtype(image, orgdtype, saturate=True)

def SaturationJitterAug(image, saturation):
    orgdtype = image.dtype
    if orgdtype in [dtypes.float16, dtypes.float32]:
        image = image
    else:
        image = tf.image.convert_image_dtype(image, dtypes.float32)

    coef = tf.convert_to_tensor([[[0.299,0.587,0.114]]])
    alpha = 1.0 - 0.4#tf.random.uniform((1,), -saturation, saturation)

    gray = image * coef
    gray = tf.reduce_sum(gray, axis=2, keepdims=True)
    gray *= (1.0-alpha)
    image *= alpha
    image += gray

    return tf.image.convert_image_dtype(image, orgdtype, saturate=True)
def HueJitterAug(image, hue):
    orgdtype = image.dtype
    if orgdtype in [dtypes.float16, dtypes.float32]:
        image = image
    else:
        image = tf.image.convert_image_dtype(image, dtypes.float32)

    tyiq = tf.convert_to_tensor([[0.299, 0.587, 0.114],
                      [0.596, -0.274, -0.321],
                      [0.211, -0.523, 0.311]])
    ityiq = tf.convert_to_tensor([[1.0, 0.956, 0.621],
                       [1.0, -0.272, -0.647],
                       [1.0, -1.107, 1.705]])
    alpha = np.random.uniform(-hue, hue)
    u = np.cos(alpha * np.pi)
    w = np.sin(alpha * np.pi)
    bt = tf.convert_to_tensor([[1.0, 0., 0.],
                    [0., u, -w],
                    [0., w, u]])
    t = tf.transpose(tf.matmul(tf.matmul(ityiq, bt), tyiq))
    image = tf.matmul(image, t)
    return tf.image.convert_image_dtype(image, orgdtype, saturate=True)

def _ColorJitter(image):
  """Random horizontal image flip."""
  # delta = np.random.uniform(-0.4, 0.4)
  # image = tf.image.adjust_brightness(image, delta=delta)
  # saturation_factor = np.random.uniform(0.6, 1.4)
  # image = tf.image.adjust_saturation(image, saturation_factor=saturation_factor)
  # contrast_factor = np.random.uniform(0.6, 1.4)
  # image = tf.image.adjust_contrast(image, contrast_factor=contrast_factor)
  param = 0.4
  image = SaturationJitterAug(image, param)
  image = BrightnessJitterAug(image, param)
  image = ContrastJitterAug(image, param)
  return image

def _Add_PCA_noise(image):
    orgdtype = image.dtype
    if orgdtype in [dtypes.float16, dtypes.float32]:
        image = image
    else:
        image = tf.image.convert_image_dtype(image, dtypes.float32)
    alphastd = 0.1
    eigval = tf.convert_to_tensor([[55.46], [4.794], [1.148]])
    eigvec = tf.convert_to_tensor([[-0.5675, 0.7192, 0.4009],
                                  [-0.5808, -0.0045, -0.8140],
                                  [-0.5836, -0.6948, 0.4203]])
    # eigval = np.array([1, 1, 1])
    # eigvec = np.ones([3,3])
    alpha = tf.random.normal((3,), 0, alphastd)
    rgb = tf.squeeze(tf.matmul(eigvec * alpha, eigval))
    image = tf.add(image, rgb/255.0)
    return tf.image.convert_image_dtype(image, orgdtype, saturate=True)


def preprocess_for_train(image_bytes, use_bfloat16, image_size=IMAGE_SIZE):
  """Preprocesses the given image for evaluation.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    use_bfloat16: `bool` for whether to use bfloat16.
    image_size: image size.

  Returns:
    A preprocessed image `Tensor`.
  """
  image = _decode_and_random_crop(image_bytes, image_size)
  image = _flip(image)
  image = tf.reshape(image, [image_size, image_size, 3])
  image = tf.image.convert_image_dtype(
      image, dtype=tf.bfloat16 if use_bfloat16 else tf.float32)
#   image = _ColorJitter(image)
#   image = _Add_PCA_noise(image)
  image = tf.subtract(image, [0.485 * 255, 0.456 * 255, 0.406 * 255])
  image = tf.divide(image, [0.229 * 255, 0.224 * 255, 0.225 * 255])
  return image


def preprocess_for_eval(image_bytes, use_bfloat16, image_size=IMAGE_SIZE):
  """Preprocesses the given image for evaluation.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    use_bfloat16: `bool` for whether to use bfloat16.
    image_size: image size.

  Returns:
    A preprocessed image `Tensor`.
  """
  image = _decode_and_center_crop(image_bytes, image_size)
  image = tf.reshape(image, [image_size, image_size, 3])
  image = tf.image.convert_image_dtype(
      image, dtype=tf.bfloat16 if use_bfloat16 else tf.float32)
  image = tf.subtract(image, [0.485 * 255, 0.456 * 255, 0.406 * 255])
  image = tf.divide(image, [0.229 * 255, 0.224 * 255, 0.225 * 255])
  return image


def preprocess_image(image_bytes, is_training=False, use_bfloat16=False,
      image_size=IMAGE_SIZE):
  """Preprocesses the given image.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    is_training: `bool` for whether the preprocessing is for training.
    use_bfloat16: `bool` for whether to use bfloat16.
    image_size: image size.

  Returns:
    A preprocessed image `Tensor` with value range of [0, 255].
  """
  if is_training:
    return preprocess_for_train(image_bytes, use_bfloat16, image_size)
  else:
    return preprocess_for_eval(image_bytes, use_bfloat16, image_size)


# from PIL import Image
# image_dir = '/media/xcq/xcqdisk/Black_Footed_Albatross_0007_796138.jpg'
# with tf.io.gfile.GFile(image_dir, 'rb') as f:
#     image_data = f.read()
# with tf.io.gfile.GFile(image_dir, 'rb') as f:
#     image_data = f.read()
# # image = preprocess_image(image_data, is_training=True)
# image = tf.image.decode_jpeg(image_data)
# # image_np = np.asarray(image.numpy()).astype(np.uint8)
# #
# # a = Image.fromarray(image_np, mode='RGB')
# #
# # a.show()
# image_noise = SaturationJitterAug(image, 0.4)
# # image_noise = _Add_PCA_noise(image)
# image_np = np.asarray(image_noise.numpy())
#
# a = Image.fromarray(image_np, mode='RGB')
#
# a.show()
# image1 = tf.image.adjust_saturation(image, 0.6)
# image_np1 = np.asarray(image1.numpy())
#
# a = Image.fromarray(image_np1, mode='RGB')
#
# a.show()
#
# image2 = tf.image.adjust_hue(image, delta=0.4)
# image_np2 = np.asarray(image2.numpy())
#
# a = Image.fromarray(image_np2, mode='RGB')
#
# a.show()
# pass
#
# pass
#
