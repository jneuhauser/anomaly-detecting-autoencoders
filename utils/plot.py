import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt


def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
        plt.title(title)


# TODO: fix me
def plot_image_and_patches(image, patches, padding='VALID'):
    if padding == 'VALID':
        round_func = math.floor
    else:
        round_func = math.ceil
    n_h = round_func(tf.shape(image)[-3] / tf.shape(patches)[-3])
    n_w = round_func(tf.shape(image)[-2] / tf.shape(patches)[-2])
    plt.figure(figsize=(1 * (4 + 1), 5))
    plt.subplot(5, 1, 1)
    plt.imshow(image)
    plt.title('original')
    plt.axis('off')
    for i, patch in enumerate(patches):
        plt.subplot(5, 5, 5 + 1 + i)
        plt.imshow(patch)
        plt.title(str(i))
        plt.axis('off')
    plt.show()


def plot_image_tuples(image_tuples, tuple_titles=None):
    rows = len(image_tuples)
    cols = len(image_tuples[0])
    fig, axs = plt.subplots(rows,cols,figsize=(15,15*rows/cols))
    for row_idx, image_tuple in enumerate(image_tuples):
        if rows > 1:
            row = axs[row_idx]
        else:
            row = axs
        if tuple_titles is not None:
            row[cols // 2].set_title(tuple_titles[row_idx])
            #axs[row_idx, cols // 2].set_title(tuple_titles[row_idx])
        for col_idx, image in enumerate(image_tuple):
            row[col_idx].imshow(image)
            #axs[row_idx, col_idx].imshow(image)
    plt.show()


def plot_greyscale_images(images, count=None, results=None):
    plot_images(images, count, results, greyscale=True)


def plot_images(images, count=None, results=None, greyscale=False):
    # From: https://www.tensorflow.org/tutorials/keras/classification#preprocess_the_data
    if count is None:
        count = len(images)
    plt.figure(figsize=(20,20))
    for i in range(count):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        if greyscale:
            plt.imshow(np.squeeze(images[i]), cmap=plt.cm.binary)
        else:
            plt.imshow(images[i])
        if results is not None:
            plt.xlabel('loss: {:.6f}, acc: {:.6f}'.format(*results[i]))
        plt.colorbar()
    plt.show()
