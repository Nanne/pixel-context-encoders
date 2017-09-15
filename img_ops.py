"""Tensorflow image operations."""
from __future__ import division
import tensorflow as tf
FLAGS = tf.app.flags.FLAGS  # parse config

def img_to_float(image):
    """Convert image to float."""
    raw_input = tf.image.convert_image_dtype(image, dtype=tf.float32)
    assertion = tf.assert_equal(tf.shape(raw_input)[2], 3,
                                message="image does not have 3 channels")
    with tf.control_dependencies([assertion]):
        raw_input = tf.identity(raw_input)

    return raw_input

def transform(image, flip, seed, scale_size, crop_size):
    """Flip and resize images."""
    r = image
    if flip:
        r = tf.image.random_flip_left_right(r, seed=seed)

    # area produces a nice downscaling,
    # but does nearest neighbor for upscaling
    # assume we're going to be doing downscaling here
    r = tf.image.resize_images(r, [scale_size, scale_size],
                               method=tf.image.ResizeMethod.AREA)

    offset = tf.cast(tf.floor(tf.random_uniform([2], 0,
                                                scale_size - crop_size + 1,
                                                seed=seed)),
                     dtype=tf.int32)
    if scale_size > crop_size:
        r = tf.image.crop_to_bounding_box(r, offset[0], offset[1],
                                          crop_size, crop_size)
    elif scale_size < crop_size:
        raise Exception("scale size cannot be less than crop size")
    return r
