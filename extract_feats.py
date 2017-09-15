from keras.applications import VGG19
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model
from keras.preprocessing import image as keras_image
import numpy as np
import skimage.transform
import h5py, json
import tensorflow as tf
import time, os

FLAGS = tf.app.flags.FLAGS

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _int64_features(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def crop_image(image, target_height=224, target_width=224):
    """Reshape image shorter and crop, keep aspect ratio."""
    width, height = image.size

    if width == height:
        resized_image = image.resize((target_width,
                                      target_height))

    elif height < width:
        w = int(width * float(target_height)/height)
        resized_image = image.resize((w, target_height))

        crop_length = int((w - target_width) / 2)
        resized_image = resized_image.crop((crop_length, 
                                           0,
                                           crop_length+target_width,
                                           target_height))
    else:
        h = int(height * float(target_width) / width)
        resized_image = image.resize((target_width, h))

        crop_length = int((h - target_height) / 2)
        resized_image = resized_image.crop((0,
                                           crop_length, 
                                           target_width,
                                           crop_length+target_height))
    return resized_image

def write_to(writer, id, image, path, tensors):
    """
    Write VGG19 layer responses to tfrecords.
    tensors should be a dict of {name : layer_response_tensor, ...}
    """
    features = {
        'id': _int64_feature(id),
        'path': _bytes_feature(path),
        'image': _bytes_feature(image.tobytes()),
        'image_size': _int64_features(list(image.size) + [3])
        }

    for name, feat in tensors.items():
        features[name] = _bytes_feature(feat.tostring())
        features[name + '_size'] = _int64_features(list(feat.shape))

    example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(example.SerializeToString())

if __name__ == '__main__':
    tf.app.flags.DEFINE_string('img_root', '/roaming/public_datasets/MS-COCO/images/val2014/',
                               'Location where original images are stored')
    tf.app.flags.DEFINE_string('record_path', '/data/cocotest.tfrecords',
                               'Directory to write the converted result to')
    tf.app.flags.DEFINE_string('extract', 'block2_conv2,block3_conv4,block4_conv4,block5_conv4',
                               """Which VGG19 layers to extract.
                               Comma seperated.""")
    tf.app.flags.DEFINE_boolean("include_top", False, "include_top for keras.VGG19 call")
    tf.app.flags.DEFINE_integer("crop_size", 0, "Size to crop to")

    # Path to the data set
    img_files = os.listdir(FLAGS.img_root)
    num_imgs = len(img_files)

    extract = [l.strip() for l in FLAGS.extract.split(',') if len(l.strip()) > 0]

    if len(extract) > 0:
        print 'Extracting:', extract

        print "Initializing model"
        # VGG19 without the last fully connected layers
        base_model = VGG19(weights='imagenet', include_top=FLAGS.include_top)


        out_layers = []
        for layer in extract:
            out_layers.append(base_model.get_layer(layer).output)

        model = Model(input=base_model.input, output=out_layers)

    writer = tf.python_io.TFRecordWriter(FLAGS.record_path)

    with open(FLAGS.record_path + ".json", 'w') as f:
        json.dump({"count": num_imgs}, f)

    print "Writing", num_imgs, "images"
    start = time.time()
    for i in range(0, num_imgs):
        print i, '\r',
        path = os.path.join(FLAGS.img_root, img_files[i])

        image = keras_image.load_img(path)
        if FLAGS.crop_size > 0:
            cropped = crop_image(image, target_height=FLAGS.crop_size, target_width=FLAGS.crop_size)
        else:
            cropped = image
        tensors = {}

        if len(extract) > 0:
            standardized = keras_image.img_to_array(cropped)
            standardized = np.expand_dims(standardized, axis=0)
            standardized = preprocess_input(standardized)

            features = model.predict(standardized)

            for j,layer in enumerate(extract):
                tensors[layer] = features[j]

        write_to(writer, i, cropped, path, tensors)

    duration = time.time() - start
    print 'Data processed in:', duration
    print num_imgs/duration, 'image/sec'
