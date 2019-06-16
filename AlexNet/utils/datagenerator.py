import tensorflow as tf
import numpy as np
import os
from tensorflow.python.framework import dtypes
from tensorflow.data import Dataset
from tensorflow.python.framework.ops import convert_to_tensor


IMAGENET_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)


class ImageDataGenerator(object):
    """Wrapper class around the new Tensorflows dataset pipeline."""

    def __init__(self, dir_file, mode, batch_size, num_classes, shuffle=True,
                 buffer_size=1000):
        """Create a new ImageDataGenerator.

        Args:
            dir_file: Path to the dataset.
            mode: Either 'training' or 'validation'. Depending on this value,
                different parsing functions will be used.
            batch_size: Number of images per batch.
            num_classes: Number of classes in the dataset.
            shuffle: Wether or not to shuffle the data in the dataset and the
                initial file list.
            buffer_size: Number of images used as buffer for TensorFlows
                shuffling of the dataset.

        """
        self.dir_file = dir_file
        self.num_classes = num_classes

        self._read_dir_file()

        self.data_size = len(self.labels)

        if shuffle:
            self._shuffle_lists()

        self.img_paths = convert_to_tensor(self.img_paths, dtype=dtypes.string)
        self.labels = convert_to_tensor(self.labels, dtype=dtypes.int32)

        data = Dataset.from_tensor_slices((self.img_paths, self.labels))

        if mode == 'training':
            data = data.map(self._parse_function_train, num_parallel_calls=4)
            data = data.prefetch(buffer_size=batch_size * 100)
        elif mode == 'inference':
            data = data.map(self._parse_function_inference,
                            num_parallel_calls=4)
            data = data.prefetch(buffer_size=batch_size * 100)
        else:
            raise ValueError("Invalid mode '%s'." % (mode))

        if shuffle:
            data = data.shuffle(buffer_size=buffer_size)

        data = data.batch(batch_size)

        self.data = data

    def _read_dir_file(self):
        """Read the content of the text file and store it into lists."""
        self.img_paths = []
        self.labels = []
        for file in os.listdir(self.dir_file):
            file_name = file.split('.')
            if file_name[0] == 'cat':
                self.img_paths.append(os.path.join(self.dir_file, file))
                self.labels.append(0)
            elif file_name[0] == 'dog':
                self.img_paths.append(os.path.join(self.dir_file, file))
                self.labels.append(1)

    def _shuffle_lists(self):
        """Conjoined shuffling of the list of paths and labels."""
        path = self.img_paths
        labels = self.labels
        permutation = np.random.permutation(self.data_size)
        self.img_paths = []
        self.labels = []
        for i in permutation:
            self.img_paths.append(path[i])
            self.labels.append(labels[i])

    def _parse_function_train(self, filename, label):
        """Input parser for samples of the training set."""
        one_hot = tf.one_hot(label, self.num_classes)

        img_string = tf.read_file(filename)
        img_decoded = tf.image.decode_jpeg(img_string, channels=3)
        img_resized = tf.image.resize_images(img_decoded, [227, 227])

        img_centered = tf.subtract(img_resized, IMAGENET_MEAN)

        # RGB -> BGR
        img_bgr = img_centered[:, :, ::-1]

        return img_bgr, one_hot

    def _parse_function_inference(self, filename, label):
        """Input parser for samples of the validation/test set."""
        one_hot = tf.one_hot(label, self.num_classes)

        img_string = tf.read_file(filename)
        img_decoded = tf.image.decode_png(img_string, channels=3)
        img_resized = tf.image.resize_images(img_decoded, [227, 227])
        img_centered = tf.subtract(img_resized, IMAGENET_MEAN)

        # RGB -> BGR
        img_bgr = img_centered[:, :, ::-1]

        return img_bgr, one_hot
