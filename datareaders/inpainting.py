import tensorflow as tf
import glob
import os
import math
from img_ops import transform, img_to_float
import random

FLAGS = tf.app.flags.FLAGS  # parse config

def feature_to_shaped(feature, shape, dtype=tf.uint8):
    shaped = tf.decode_raw(feature, dtype)
    shaped = tf.squeeze(tf.reshape(shaped, tf.cast(shape, tf.int32)))
    return shaped

def read_record(filename_queue, aux=False, augment=True):
    """
    Read fromTFrecords containing vgg features.

    based on http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/
    """
    # Initialize reader
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # Parse TFRecords
    features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
            'id': tf.FixedLenFeature([1], tf.int64),
            'path': tf.FixedLenFeature([], tf.string),
            'image': tf.FixedLenFeature([], tf.string),
            'image_size': tf.FixedLenFeature([3], tf.int64),
        })

    path = features['path']
    tensors = [path]

    # Reshape byte-strings to original shape
    image = feature_to_shaped(features['image'], features['image_size'], dtype=tf.uint8)
    image.set_shape((256,256,3))

    pre_image = preprocess_input(image)

    im_size = FLAGS.im_size
    mask_size = FLAGS.mask_size
    mask_width = mask_size
    mask_height = mask_size
    channels = 3

    pre_image = tf.image.resize_images(pre_image, (im_size, im_size))
    pre_image.set_shape((im_size,im_size,3))
    if augment:
        pre_image = tf.image.random_flip_left_right(pre_image, seed=FLAGS.seed)

    # spread is how much remains of the image, in either dimension
    spread_h = tf.constant(int(im_size - mask_size))
    spread_w = tf.constant(int(im_size - mask_size))
    if FLAGS.random_mask:
        left = tf.random_uniform((1,), maxval=spread_w, dtype=tf.int32)
        top = tf.random_uniform((1,), maxval=spread_h, dtype=tf.int32)
    elif FLAGS.side_mask:
        mask_height = im_size
        spread_h = tf.constant(int(im_size - mask_height))
        spread_w = tf.constant(int(im_size - mask_width))
        left = tf.constant([im_size-mask_width])
        top = tf.constant([0])
    else: # center mask
        left = tf.constant([(im_size-mask_width)/2])
        top = tf.constant([(im_size-mask_height)/2])

    depth_pad = tf.zeros((2,1), dtype=tf.int32)
    padding = tf.squeeze(tf.stack([[top,spread_h-top], [left,spread_w-left], depth_pad]))

    mask = tf.ones((mask_height,mask_width,channels))
    mask = tf.pad(mask, padding)
    mask.set_shape((im_size,im_size,channels))

    if FLAGS.invert_mask:
        mask = tf.abs(mask - 1)

    if FLAGS.overlap > 0:
        overlap = FLAGS.overlap
        if FLAGS.invert_mask:
            overlap_mask = tf.ones((mask_height+2*overlap, mask_width+2*overlap, channels))
            overlap_mask = tf.pad(overlap_mask, 
                    tf.squeeze(tf.stack([[left-overlap,spread-left-overlap], 
                                        [top-overlap,spread-top-overlap], 
                                        depth_pad])))
            overlap_mask = tf.abs(overlap_mask - 1)
        elif FLAGS.side_mask: # Only have overlap on left side for side_mask
            overlap_mask = tf.ones((mask_height, mask_width-overlap, channels))
            overlap_mask = tf.pad(overlap_mask, tf.stack([[0, 0], [overlap, 0], tf.squeeze(depth_pad)]))
            overlap_mask = tf.pad(overlap_mask, padding)
        else:
            overlap_mask = tf.ones((mask_height-2*overlap, mask_width-2*overlap, channels))
            overlap_mask = tf.pad(overlap_mask, tf.stack([[overlap, overlap], [overlap, overlap], tf.squeeze(depth_pad)]))
            overlap_mask = tf.pad(overlap_mask, padding)
        overlap_mask.set_shape((im_size,im_size,channels))
    else:
        overlap_mask = mask

    #TODO could optimise to not store input, only full image and mask
    input = tf.multiply(pre_image, 1 - overlap_mask)
    tensors.append(input) # input

    """if FLAGS.invert_mask or FLAGS.architecture == "PCE":
        target = tf.multiply(pre_image, mask)
    else:
        bool_mask = tf.equal(mask, 1)
        target = tf.boolean_mask(pre_image, bool_mask)
        target = tf.reshape(target, (mask_height,mask_width,channels))
        target.set_shape((mask_height,mask_width,channels))"""

    tensors.append(pre_image) # target

    tensors.append(mask)

    return tensors

def preprocess_input(image):
    """ uint8 [0,255] -> float32 [-1, 1] """
    with tf.name_scope("preprocess_input"):
        image = tf.image.convert_image_dtype(image, tf.float32, saturate=True)
        return image * 2 - 1

def deprocess_input(image):
    """float32 [-1, 1] => uint8 [0, 255]."""
    with tf.name_scope("deprocess_input"):
        image = (image + 1) / 2
        return tf.image.convert_image_dtype(image, tf.uint8, saturate=True)

def preprocess_output(image):
    """ uint8 [0,255] -> float32 [-1, 1] """
    with tf.name_scope("preprocess_output"):
        image = tf.image.convert_image_dtype(image, tf.float32, saturate=True)
        return image * 2 - 1

def deprocess_output(image):
    """float32 [-1, 1] => uint8 [0, 255]."""
    with tf.name_scope("deprocess_output"):
        image = (image + 1) / 2
        return tf.image.convert_image_dtype(image, tf.uint8, saturate=True)

def reconstruct_inpaint(images, outputs, masks):
    with tf.name_scope("reconstruct_inpaint"):

        if (FLAGS.invert_mask and not FLAGS.side_mask) or FLAGS.architecture == "PCE": 
            padded_output = tf.image.convert_image_dtype(outputs, tf.float32, saturate=False)
            padded_output = tf.multiply(padded_output, masks)
        else:
            H = tf.shape(outputs)[1]
            W = tf.shape(outputs)[2]
            D = tf.shape(outputs)[3]
            spread_H = tf.shape(images)[1] - H
            spread_W = tf.shape(images)[2] - W

            # find top left of masked region
            all_indices = tf.slice(tf.where(tf.equal(masks,1)), [0,0], [-1, 3])
            indices = tf.strided_slice(all_indices, [0, 0], [H*W*D*FLAGS.batch_size, 3], [H*W*D, 1])
            top = tf.cast(indices[:,1], tf.int32)
            left = tf.cast(indices[:,2], tf.int32)
            # pad output such that it lines up with masked region
            depth_pad = tf.zeros((2,FLAGS.batch_size), dtype=tf.int32)
            padding = tf.stack([tf.reshape([top,spread_H-top], (2,FLAGS.batch_size)), tf.reshape([left,spread_W-left], (2,FLAGS.batch_size)), depth_pad], axis=2)
            padding = tf.transpose(padding, (1,2,0))

            padded_output = tf.map_fn(lambda (i,p): tf.pad(i,p), (outputs, padding), dtype=images.dtype)
            padded_output.set_shape(images.shape)
            padded_output = tf.image.convert_image_dtype(padded_output, tf.float32, saturate=False)

        convert_back = False
        if images.dtype == tf.uint8:
            convert_back = True
            images = tf.image.convert_image_dtype(images, tf.float32, saturate=False)

        i = tf.multiply(images, 1-masks)
        recon = tf.add(padded_output, i)

        if convert_back:
            recon = tf.image.convert_image_dtype(recon, tf.uint8, saturate=True)

        return recon