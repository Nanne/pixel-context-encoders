from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import json
import random
import math
import time
import cfg, util 
# load some options from the checkpoint
util.restore_flags(verbose=False)
import dataprovider
from create_model import create_model

FLAGS = tf.app.flags.FLAGS  # parse config
CROP_SIZE = 256
FLAGS.crop_size = CROP_SIZE

def main(argv=None):    # pylint: disable=unused-argument
    """Evaluate model and store results + visuals (latter only if specified)."""
    tf.set_random_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)

    if FLAGS.checkpoint is None:
        raise Exception("checkpoint required for evaluation")

    if not os.path.exists(FLAGS.checkpoint):
        raise Exception("valid checkpoint required for evaluation")

    if FLAGS.visuals_dir != None:
        # Store files in FLAGS.output/FLAGS.checkpoint_dir
        # create paths if needed
        if not os.path.exists(FLAGS.visuals_dir):
            os.makedirs(FLAGS.visuals_dir)
        visuals_dir = os.path.join(FLAGS.visuals_dir, FLAGS.checkpoint)
        if not os.path.exists(visuals_dir):
            os.makedirs(visuals_dir)

    # load some options from the checkpoint
    util.restore_flags()
    # disable these features in test mode
    FLAGS.scale_size = CROP_SIZE
    FLAGS.flip = False

    examples = dataprovider.load_records(train=False)

    # Retrieve data specific function deprocess
    # It deprocesss output image
    deprocess_input = examples.deprocess_input
    deprocess_output = examples.deprocess_output

    print("examples count = %d" % examples.count)
    model = create_model(examples)

    with tf.name_scope("images_summary"):
        deprocessed_images = tf.cast(deprocess_input(examples.inputs), examples.masks.dtype) * (1 - examples.masks)
        # make masked area white instead of black
        deprocessed_images += (tf.ones_like(deprocessed_images) * 255) * examples.masks
        deprocessed_images = tf.cast(deprocessed_images, tf.uint8)

    with tf.name_scope("targets_summary"):
        deprocessed_targets = deprocess_output(examples.targets)

    with tf.name_scope("outputs_summary"):
        deprocessed_outputs = deprocess_output(model.outputs)

    if FLAGS.visuals_dir != None:
        with tf.name_scope("encode_images"):
            display_fetches = {
                "paths": examples.paths,
                "images": tf.map_fn(tf.image.encode_png, deprocessed_images,
                                    dtype=tf.string, name="input_pngs"),
                "targets": tf.map_fn(tf.image.encode_png, deprocessed_targets,
                                     dtype=tf.string, name="target_pngs"),
                "outputs": tf.map_fn(tf.image.encode_png, deprocessed_outputs,
                                     dtype=tf.string, name="output_pngs"),
            }


    if FLAGS.decoder:
        with tf.name_scope("squared_error"):
            l2_error = tf.pow(tf.to_float(deprocessed_targets) - tf.to_float(deprocessed_outputs), 2)

            sse = tf.reduce_sum(l2_error)
            pixels = tf.reduce_prod(l2_error.shape)

    saver = tf.train.Saver(max_to_keep=1)

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    sv = tf.train.Supervisor(logdir=None, save_summaries_secs=0, saver=None)
    sess_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    with sv.managed_session(config=sess_config) as sess:
        print("parameter_count =", sess.run(parameter_count))

        print("loading model from checkpoint")
        checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint)
        saver.restore(sess, checkpoint)

        fetches = {
            "global_step": sv.global_step,
            "loss": model.gen_loss_content
        }

        if FLAGS.visuals_dir:
            fetches['display'] = display_fetches

        if FLAGS.decoder:
            fetches['sse'] = sse
            fetches['pixels'] = pixels

        print("Loaded model, evaluating")

        total_sse = 0
        total_pixels = 0
        losses = []

        for step in range(examples.steps_per_epoch):
        #for step in range(10):

            if step % 50 == 0:
                print(step, '/', examples.steps_per_epoch)

            results = sess.run(fetches)

            losses.append(results['loss'])

            if FLAGS.visuals_dir != None:
                filesets = util.save_images(results["display"],
                                            visuals_dir)
            if FLAGS.decoder:
                total_sse += results['sse']
                total_pixels += results['pixels']

        if FLAGS.decoder:
            MSE = total_sse / total_pixels
            RMSE = np.sqrt(MSE)
            PIXEL_MAX = 255
            PSNR = 20 * np.log10(PIXEL_MAX / RMSE)
            mean_loss = np.mean(losses)

            print("loss:\t",  mean_loss)
            print("MSE:\t",  MSE)
            print("RMSE:\t", RMSE)
            print("PSNR:\t", PSNR)
        else:
            print("Done!")


if __name__ == '__main__':
    tf.app.run()
