import tensorflow as tf
import glob, os.path
import os
import math
from img_ops import transform, img_to_float
import random

FLAGS = tf.app.flags.FLAGS  # parse config

def feature_to_shaped(feature, shape, dtype=tf.uint8):
    shaped = tf.decode_raw(feature, dtype)
    shaped = tf.squeeze(tf.reshape(shaped, tf.cast(shape, tf.int32)))
    return shaped

def read_record(filename_queue, aux=False, augment=False):

    images = [i for i in glob.glob('demo/*.png') if not i.endswith('mask.png')]
    FLAGS.num_samples = len(images)
    FLAGS.batch_size = min(50, len(images))

    masks = []
    for im in images:
        n, ext = os.path.splitext(os.path.basename(im))
        custom_mask = os.path.join('demo/', n + "_mask" + ext)
        if os.path.exists(custom_mask):
            masks.append(custom_mask)
        else:
            masks.append('demo/mask.png')
    assert len(images) == len(masks)

    filename_queue = tf.train.string_input_producer(images, shuffle=False)
    reader = tf.WholeFileReader()
    path, value = reader.read(filename_queue)
    image = tf.image.decode_png(value)
    image = image[:,:,:3] # get rid of potential alpha channel
    pre_image = preprocess_input(image)
    # take the top left 256x256 pixels
    pre_image = tf.image.crop_to_bounding_box(pre_image, 0, 0, 256, 256)
    pre_image.set_shape((256,256,3))

    mask_queue = tf.train.string_input_producer(masks, shuffle=False)
    mask_reader = tf.WholeFileReader()
    _, mask_value = reader.read(mask_queue)
    mask = tf.image.decode_png(mask_value)
    mask = mask[:,:,:3]
    mask = tf.round(tf.cast(mask, tf.float32) / 255)
    mask = tf.image.crop_to_bounding_box(mask, 0, 0, 256, 256)
    mask.set_shape((256,256,3))
    
    input = tf.multiply(pre_image, 1 - mask)

    tensors = [[path]]
    tensors.append(input) # input
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

        padded_output = tf.image.convert_image_dtype(outputs, tf.float32, saturate=False)
        padded_output = tf.multiply(padded_output, masks)

        convert_back = False
        if images.dtype == tf.uint8:
            convert_back = True
            images = tf.image.convert_image_dtype(images, tf.float32, saturate=False)

        i = tf.multiply(images, 1-masks)
        recon = tf.add(padded_output, i)

        if convert_back:
            recon = tf.image.convert_image_dtype(recon, tf.uint8, saturate=True)

        return recon
