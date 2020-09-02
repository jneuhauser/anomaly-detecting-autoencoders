from __future__ import absolute_import

import os

import tensorflow as tf
import numpy as np

# pylint: disable=relative-beyond-top-level
from .dataset_utils import normalize_dataset, normalize_images, merge_and_split_train_test, reshape_dataset


def get_dataset_mnist(normalize=True, reshape=True):
    """
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/datasets/mnist.py
    """
    dataset = tf.keras.datasets.mnist.load_data()
    if normalize:
        dataset = normalize_dataset(dataset)
    if reshape:
        dataset = reshape_dataset(dataset, colors=1)
    return dataset


def get_dataset_fashion_mnist(normalize=True, reshape=True):
    """
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/datasets/fashion_mnist.py
    """
    dataset = tf.keras.datasets.fashion_mnist.load_data()
    if normalize:
        dataset = normalize_dataset(dataset)
    if reshape:
        dataset = reshape_dataset(dataset, colors=1)
    return dataset


def get_dataset_cifar10(normalize=True, reshape=True):
    """
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/datasets/cifar10.py
    """
    dataset = tf.keras.datasets.cifar10.load_data()
    if normalize:
        dataset = normalize_dataset(dataset)
    if reshape:
        dataset = reshape_dataset(dataset, colors=3)
    return dataset


def get_dataset_cifar100(normalize=True, reshape=True):
    """
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/datasets/cifar100.py
    """
    dataset = tf.keras.datasets.cifar100.load_data()
    if normalize:
        dataset = normalize_dataset(dataset)
    if reshape:
        dataset = reshape_dataset(dataset, colors=3)
    return dataset


def get_dataset_stl10(normalize=True, reshape=True, test_percentage=10, unlabeled=False):
    """
    http://ai.stanford.edu/~acoates/stl10/
    """
    dirname = os.path.join('datasets', 'stl10')
    base = 'http://ai.stanford.edu/~acoates/stl10/'
    file = 'stl10_binary.tar.gz'

    lpath = '/home/ec2-user/.keras/datasets/stl10/stl10_binary.tar.gz'
    # assume extracted files exist if archive file exists
    if not os.path.isfile(lpath):
        lpath = tf.keras.utils.get_file(file, origin=base + file, cache_subdir=dirname, extract=True)

    def _read_images_from_file(f):
        images = np.fromfile(f, dtype=np.uint8)
        images = np.reshape(images, (-1, 3, 96, 96))
        images = np.transpose(images, (0, 3, 2, 1))
        return images

    def _read_labels_from_file(f):
        labels = np.fromfile(f, dtype=np.uint8)
        return labels

    for root, dirs, files in os.walk(os.path.dirname(lpath)):
        for file in files:
            file_path = os.path.join(root, file)
            if file_path.endswith('train_X.bin'):
                train_x = _read_images_from_file(file_path)
            elif file_path.endswith('train_y.bin'):
                train_y = _read_labels_from_file(file_path)
            elif file_path.endswith('test_X.bin'):
                test_x = _read_images_from_file(file_path)
            elif file_path.endswith('test_y.bin'):
                test_y = _read_labels_from_file(file_path)
            elif unlabeled and file_path.endswith('unlabeled_X.bin'):
                unlabeled_x = _read_images_from_file(file_path)
            # only keep compressed file
            #if not file_path.endswith('.tar.gz'):
            #    os.remove(file_path)

    if test_percentage is not None:
        train_x, test_x = merge_and_split_train_test(train_x, test_x, test_percentage=test_percentage)
        train_y, test_y = merge_and_split_train_test(train_y, test_y, test_percentage=test_percentage)

    if normalize:
        train_x, test_x = normalize_images(train_x, test_x)
    elif normalize and unlabeled:
        train_x, test_x, unlabeled_x = normalize_images(train_x, test_x, unlabeled_x)

    if not unlabeled:
        return (train_x, train_y), (test_x, test_y)
    else:
        return (train_x, train_y), (test_x, test_y), unlabeled_x

