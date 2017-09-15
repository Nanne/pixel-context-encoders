import tensorflow as tf
import json
import os


tf.app.flags.DEFINE_string("input_dir", '', "path to tfrecord containing images")
tf.app.flags.DEFINE_boolean("records", True,
                            "Use TFRecords. If true '--input_dir' \
                             must be a path to the record")
tf.app.flags.DEFINE_integer("num_samples", None,
                            "When using TFrecords the number of samples have to \
                             be calculated at runtime. Providing a value for \
                             --num_samples bypasses this computation.")
tf.app.flags.DEFINE_string("output_dir", None, "where to put output files")
tf.app.flags.DEFINE_string("visuals_dir", None, "where to put generated image files")
tf.app.flags.DEFINE_string("checkpoint", None,
                           "directory with checkpoint to resume training \
                            from or use for testing")

tf.app.flags.DEFINE_boolean("decoder", True,
                            'Attach top-half (decoder) of U-net. \
                             If True trains with --content_loss'  )
tf.app.flags.DEFINE_boolean("discriminator", True,
                            "Add or remove discriminator. \
                             If True train with GAN loss on top \
                             of content loss.")

tf.app.flags.DEFINE_integer("overlap", 0,
                            "Overlapping edge for inpainting")
#TODO change overlap_weight to scalar
tf.app.flags.DEFINE_boolean("overlap_weight", False,
                            """Add extra weight to overlapping region. Set to false for eval when trained with overlap=0""")
tf.app.flags.DEFINE_integer("mask_size", 64, "Size of mask for inpainting")
tf.app.flags.DEFINE_boolean("random_mask", False,
                            """Randomise the mask position?""")
tf.app.flags.DEFINE_boolean("invert_mask", False,
                            """Invert the boolean mask?""")
tf.app.flags.DEFINE_boolean("side_mask", False,
                            """Mask the `mask_size' pixels on the right of the image. Invert for left""")

tf.app.flags.DEFINE_string("architecture", 'PCE', "Choose 'PCE'")
tf.app.flags.DEFINE_string("datareader", 'inpainting', "Choose 'inpainting''")

tf.app.flags.DEFINE_integer("ngf", 128,
                            "number of generator filters in first conv layer \
                             the decoder.")
tf.app.flags.DEFINE_integer("--ndf", 64,
                            "number of discriminator filters \
                             in first conv layer of the discriminator")

tf.app.flags.DEFINE_float("bat_loss", 0,
                          "Weight for Bhattaccarya loss")
tf.app.flags.DEFINE_float("tv_loss", 0,
                            "Weight for total variation loss")
tf.app.flags.DEFINE_float("l1_loss", 1,
                            "Weight for L1 loss")
tf.app.flags.DEFINE_float("l2_loss", 0,
                           "Weight for L2 loss")
tf.app.flags.DEFINE_float("content_weight", 0.999,
                          "weight on content term for generator gradient")
tf.app.flags.DEFINE_float("aux_weight", 10.0,
                          "weight on aux term for generator gradient")
tf.app.flags.DEFINE_float("gan_weight", 0.001,
                          "weight on GAN term for generator gradient")


tf.app.flags.DEFINE_integer("max_epochs", 0, "number of training epochs")
tf.app.flags.DEFINE_integer("summary_freq", 100,
                            "update summaries every summary_freq steps")
tf.app.flags.DEFINE_integer("progress_freq", 50,
                            "display progress every progress_freq steps")
# to get tracing working on GPU, LD_LIBRARY_PATH may need to be modified:
# LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/extras/CUPTI/lib64
tf.app.flags.DEFINE_integer("trace_freq", 0,
                            "trace execution every trace_freq steps")
tf.app.flags.DEFINE_integer("display_freq", 0,
                            "write current training images \
                             every display_freq steps")
tf.app.flags.DEFINE_integer("save_freq", 5000,
                            "save model every save_freq steps, 0 to disable")

tf.app.flags.DEFINE_boolean("flip", True, "flip images horizontally")
tf.app.flags.DEFINE_integer("im_size", 128, "Resize images to this size before processing")

tf.app.flags.DEFINE_integer("batch_size", 1, "number of images in batch")
tf.app.flags.DEFINE_float("lr", 0.0002, "initial learning rate for adam")
tf.app.flags.DEFINE_float("beta1", 0.5, "momentum term of adam")

tf.app.flags.DEFINE_integer("seed", 1860795210, "Random seed")

"""Restore options from checkpoint/options.json."""
restore_flags = {"ngf", "ndf",
           "gan_weight", "content_weight", "lr", "beta1",
           "trace_freq", "summary_freq", "aux", "aux_weight",
           "num_classes", "discriminator", "overlap",
           "decoder", 'bat_loss', 'l1_loss', 'l2_loss',
           'architecture', 'datareader', 'pretrained', 
           'im_size', 'mask_size', 'side_mask',
           'inpainting', 'tv_loss', 'random_mask', 'invert_mask'}
