"""
Script to convert dataset .

"""

import math
import os
import random
import tensorflow as tf
import argparse




TRAINING_SHARDS = 1000
VALIDATION_SHARDS = 100

TRAINING_DIRECTORY = 'train'
VALIDATION_DIRECTORY = 'validation'

parser = argparse.ArgumentParser(description='Convert imagenet dataset to TFRECORDS')
parser.add_argument('--dataset', metavar='DIR', default=None,
                    help='path to dataset')


def _check_or_create_dir(directory):
  """Check if directory exists otherwise create it."""
  if not os.path.exists(directory):
    os.makedirs(directory)


def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(filename, image_buffer, label, height, width):
  """Build an Example proto for an example.

  Args:
    filename: string, path to an image file, e.g., '/path/to/example.JPG'
    image_buffer: string, JPEG encoding of RGB image
    label: integer, identifier for the ground truth for the network
    height: integer, image height in pixels
    width: integer, image width in pixels
  Returns:
    Example proto
  """
  image_format = b'JPEG'

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': _int64_feature(height),
      'image/width': _int64_feature(width),
      'image/class/label': _int64_feature(label),
      'image/format': _bytes_feature(image_format),
      'image/encoded': _bytes_feature(image_buffer)}))
  return example


def _is_png(filename):
  """Determine if a file contains a PNG format image.

  Args:
    filename: string, path of the image file.

  Returns:
    boolean indicating if the image is a PNG.
  """
  # File list from:
  # https://github.com/cytsai/ilsvrc-cmyk-image-list
  return 'n02105855_2933.JPEG' in filename


def _is_cmyk(filename):
  """Determine if file contains a CMYK JPEG format image.

  Args:
    filename: string, path of the image file.

  Returns:
    boolean indicating if the image is a JPEG encoded with CMYK color space.
  """
  # File list from:
  # https://github.com/cytsai/ilsvrc-cmyk-image-list
  blacklist = set(['n01739381_1309.JPEG', 'n02077923_14822.JPEG',
                   'n02447366_23489.JPEG', 'n02492035_15739.JPEG',
                   'n02747177_10752.JPEG', 'n03018349_4028.JPEG',
                   'n03062245_4620.JPEG', 'n03347037_9675.JPEG',
                   'n03467068_12171.JPEG', 'n03529860_11437.JPEG',
                   'n03544143_17228.JPEG', 'n03633091_5218.JPEG',
                   'n03710637_5125.JPEG', 'n03961711_5286.JPEG',
                   'n04033995_2932.JPEG', 'n04258138_17003.JPEG',
                   'n04264628_27969.JPEG', 'n04336792_7448.JPEG',
                   'n04371774_5854.JPEG', 'n04596742_4225.JPEG',
                   'n07583066_647.JPEG', 'n13037406_4650.JPEG'])
  return os.path.basename(filename) in blacklist


class ImageCoder(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
      return

  def png_to_jpeg(self, image_data):
      image = tf.image.decode_png(image_data, channels=3)
      _png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100).numpy()
      return _png_to_jpeg

  def cmyk_to_rgb(self, image_data):

      image = tf.image.decode_jpeg(image_data, channels=0)
      _cmyk_to_rgb = tf.image.encode_jpeg(image, format='rgb', quality=100).numpy()
      return _cmyk_to_rgb

  def decode_jpeg(self, image_data):
      _decode_jpeg = tf.image.decode_jpeg(image_data, channels=3)
      assert len(_decode_jpeg.shape) == 3
      assert _decode_jpeg.shape[2] == 3
      return _decode_jpeg


def _process_image(filename, coder):
  """Process a single image file.

  Args:
    filename: string, path to an image file e.g., '/path/to/example.JPG'.
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
  Returns:
    image_buffer: string, JPEG encoding of RGB image.
    height: integer, image height in pixels.
    width: integer, image width in pixels.
  """
  # Read the image file.
  with tf.io.gfile.GFile(filename, 'rb') as f:
    image_data = f.read()

  # Clean the dirty data.
  if _is_png(filename):
    # 1 image is a PNG.
    # tf.logging.info('Converting PNG to JPEG for %s' % filename)
    image_data = coder.png_to_jpeg(image_data)
  elif _is_cmyk(filename):
    # 22 JPEG images are in CMYK colorspace.
    # tf.logging.info('Converting CMYK to RGB for %s' % filename)
    image_data = coder.cmyk_to_rgb(image_data)


  # Decode the RGB JPEG.
  image = coder.decode_jpeg(image_data)

  # Check that image converted to RGB
  assert len(image.shape) == 3
  height = image.shape[0]
  width = image.shape[1]
  assert image.shape[2] == 3

  return image_data, height, width


def _process_image_files_batch(coder, output_file, filenames, labels):
  """Processes and saves list of images as TFRecords.

  Args:
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
    output_file: string, unique identifier specifying the data set
    filenames: list of strings; each string is a path to an image file
    labels: labels
  """
  writer = tf.io.TFRecordWriter(output_file)

  for i in range(len(filenames)):
    filename = filenames[i]
    label = labels[i]
    image_buffer, height, width = _process_image(filename, coder)
    example = _convert_to_example(filename, image_buffer, label, height, width)
    writer.write(example.SerializeToString())

  writer.close()


def _process_dataset(filenames, labels, output_directory, prefix, num_shards):
  """Processes and saves list of images as TFRecords.

  Args:
    filenames: list of strings; each string is a path to an image file
    labels: labels
    output_directory: path where output files should be created
    prefix: string; prefix for each file
    num_shards: number of chucks to split the filenames into

  Returns:
    files: list of tf-record filepaths created from processing the dataset.
  """
  _check_or_create_dir(output_directory)
  chunksize = int(math.ceil(len(filenames) / num_shards))
  coder = ImageCoder()

  files = []

  for shard in range(num_shards):
    chunk_files = filenames[shard * chunksize : (shard + 1) * chunksize]
    chunk_labels = labels[shard * chunksize : (shard + 1) * chunksize]
    output_file = os.path.join(
        output_directory, '%s-%.5d-of-%.5d.tfrecords' % (prefix, shard, num_shards))
    _process_image_files_batch(coder, output_file, chunk_files, chunk_labels)
    # tf.logging.info('Finished writing file: %s' % output_file)
    files.append(output_file)
  return files


def convert_to_tf_records(raw_data_dir):
    """Convert the dataset into TF-Record dumps."""

    # Glob all the training files
    training_files = []
    training_labels = []
    train_dir = os.path.join(raw_data_dir, 'train')
    train_species = sorted(os.listdir(train_dir))
    for i in range(len(train_species)):
        s_dir = os.path.join(train_dir, train_species[i])
        # 遍历目录下的所有图片
        for filename in os.listdir(s_dir):
            # 获取文件的路径
            training_files.append(os.path.join(s_dir, filename))
            training_labels.append(int(i+1))

    # Shuffle training records to ensure we are distributing classes
    # across the batches.
    training_shuffle_idx = list(zip(training_files, training_labels))
    random.shuffle(training_shuffle_idx)
    training_files, training_labels = zip(*training_shuffle_idx)

    # Glob all the validation files
    validation_files = []
    validation_labels = []
    val_dir = os.path.join(raw_data_dir, 'val')
    val_species = sorted(os.listdir(val_dir))
    for i in range(len(val_species)):
        s_dir = os.path.join(val_dir, val_species[i])
        # 遍历目录下的所有图片
        for filename in os.listdir(s_dir):
            # 获取文件的路径
            validation_files.append(os.path.join(s_dir, filename))
            validation_labels.append(int(i + 1))

    # Create training data
    # tf.logging.info('Processing the training data.')
    training_records = _process_dataset(
        training_files, training_labels,
        os.path.join(raw_data_dir, 'tfrecords'),
        TRAINING_DIRECTORY, TRAINING_SHARDS)

    # Create validation data
    # tf.logging.info('Processing the validation data.')
    validation_records = _process_dataset(
        validation_files, validation_labels,
        os.path.join(raw_data_dir, 'tfrecords'),
        VALIDATION_DIRECTORY, VALIDATION_SHARDS)

    return training_records, validation_records


if __name__ == '__main__':
    global args
    args = parser.parse_args()
    convert_to_tf_records(args.dataset)
