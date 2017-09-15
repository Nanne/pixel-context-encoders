import tensorflow as tf
import glob
import os
import math
import random
from util import to_namedtuple
import json

FLAGS = tf.app.flags.FLAGS  # parse config

if FLAGS.datareader == "inpainting":
    from datareaders.inpainting import read_record, deprocess_input, deprocess_output, reconstruct_inpaint
elif FLAGS.datareader == "extrapolate":
    from datareaders.extrapolate import read_record, deprocess_input, deprocess_output, reconstruct_inpaint
elif FLAGS.datareader == "demo":
    from datareaders.demo import read_record, deprocess_input, deprocess_output, reconstruct_inpaint
else:
    raise ValueError('Unknown dataset option: ' + FLAGS.datareader)

def getsize(filename):
    jsfile = filename + ".json"
    if tf.gfile.Exists(jsfile):
        with open(jsfile, 'r') as f:
            N = json.load(f)['count']
    else:
        N = 0
        print "Obtaining sample count:"
        for record in tf.python_io.tf_record_iterator(filename):
            N += 1
            print N, '\r',
        with open(jsfile, 'w') as f:
            f.write(json.dumps({'count': N}))
    return N

def load_records(train=True):
    """Imports a read_record function from the module corresponding to the dataset
    and reads records, in batches, from the dataset."""

    e = {} # container for examples
    records_array = [p.strip() for p in FLAGS.input_dir.split(",")]

    if len(records_array) > 0 and records_array != ['']:
        for records_file in records_array:
            if not os.path.exists(records_file):
                raise Exception("Path to record doesn't exist", records_file)

        with tf.name_scope("load_images"):
            filename_queue = tf.train.string_input_producer(records_array)
            tensors = read_record(filename_queue, augment=train)
    else:
        # No records provided, assuming that we're using a datareader that
        # knows where the data is
        with tf.name_scope("load_images"):
            tensors = read_record('', augment=train)

    # Need to loop through all records and count if sample count not given
    if not FLAGS.num_samples:
        num_samples = 0
        for records_file in records_array:
            num_samples += getsize(records_file)
    else:
        num_samples = FLAGS.num_samples

    if train:
        batch = tf.train.shuffle_batch(tensors,
                                       batch_size=FLAGS.batch_size,
                                       capacity=FLAGS.batch_size*100,
                                       min_after_dequeue=FLAGS.batch_size*50,
                                       num_threads=16)
    else:
        batch = tf.train.batch(tensors, batch_size=FLAGS.batch_size)

    batch = list(batch)
    steps_per_epoch = max(1, int(math.ceil(num_samples / FLAGS.batch_size)))
    e["steps_per_epoch"], e["count"]  = steps_per_epoch, num_samples

    paths_batch = batch.pop(0)
    e["paths"] = paths_batch

    inputs_batch = batch.pop(0)
    e["inputs"] = inputs_batch

    if FLAGS.decoder:
        targets_batch = batch.pop(0)
        e["targets"] = targets_batch
    else:
        e['targets'] = None

    e["masks"] = batch.pop(0)
    e["reconstruct_inpaint"] = reconstruct_inpaint

    e["deprocess_input"] = deprocess_input
    e["deprocess_output"] = deprocess_output

    examples = to_namedtuple(e, "Examples")
    return examples
