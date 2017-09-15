
"""Couple of layers implemented in tensorflow."""
import tensorflow as tf

if tf.__version__ == "0.12.1":
    concatenate = tf.concat_v2
else:
    concatenate = tf.concat

if tf.__version__ == "0.12.1":
    init_zeros = tf.zeros_initializer
else:
    init_zeros = tf.zeros_initializer()

def dense(batch_input, out_dim):
    """Add dense layer to the graph."""
    with tf.name_scope('dense'):
        in_dim = batch_input.get_shape()[1]
        weights = tf.get_variable("weights", [in_dim, out_dim],
                                  dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.Variable(tf.zeros([out_dim]),
                             name='biases')
        activation = tf.matmul(batch_input, weights) + biases
        return activation

# based on https://github.com/jazzsaxmafia/Inpainting/blob/master/src/model.py
def channel_wise_dense(batch_input): # bottom: (7x7x512)
    """Add channel wise dense layer to the graph """
    with tf.variable_scope('channel_wise_dense'):
        _, width, height, n_feat_map = batch_input.get_shape().as_list()
        input_reshape = tf.reshape( batch_input, [-1, width*height, n_feat_map] )
        input_transpose = tf.transpose( input_reshape, [2,0,1] )

        W = tf.get_variable(
                "weights",
                shape=[n_feat_map,width*height, width*height], # (512,49,49)
                initializer=tf.contrib.layers.xavier_initializer())
        output = tf.matmul(input_transpose, W)

        output_transpose = tf.transpose(output, [1,2,0])
        output_reshape = tf.reshape( output_transpose, [-1, height, width, n_feat_map] )

        return output_reshape

def conv(batch_input, out_channels, stride, filter_size=3, padding="SAME"):
    """
    Add convolutional layer to the graph.

    [batch, in_height, in_width, in_channels],
    [filter_width, filter_height, in_channels, out_channels]
    => [batch, out_height, out_width, out_channels]
    """
    with tf.variable_scope("conv"):
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter", [filter_size, filter_size,
                                            in_channels,
                                            out_channels],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())


        conv = tf.nn.conv2d(batch_input, filter, [1, stride, stride, 1],
                            padding=padding)
        return conv

class Identity():
    def __init__(self, dtype=tf.float32):
        self.dtype = tf.as_dtype(dtype)

    def __call__(self, shape, dtype=None, partition_info=None):
        if dtype is None:
          dtype = self.dtype

        if len(shape) == 2 and shape[0] == shape[1]:
            return tf.diag(tf.ones(shape[0], dtype))
        elif len(shape) == 4 and shape[2] == shape[3]:
            q = tf.reshape(tf.diag(tf.ones(shape[2], dtype)), (1,1,shape[2], shape[3]))
            h = int((shape[0]-1)/2)
            w = int((shape[1]-1)/2)
            padding = [(h,h), (w,w), (0,0), (0,0)]
            return tf.pad(q, padding)
        else:
            raise

    def get_config(self):
        return {"dtype": self.dtype.name}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

identity_initializer = Identity

def dilated_conv(batch_input, out_channels, rate, filter_size=3, padding="SAME"):
    """
    Add dilated convolutional layer to the graph.

    [batch, in_height, in_width, in_channels],
    [filter_width, filter_height, in_channels, out_channels]
    => [batch, out_height, out_width, out_channels]
    """
    with tf.variable_scope("dilated_conv"):
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter", [filter_size, filter_size,
                                            in_channels,
                                            out_channels],
                                 dtype=tf.float32,
                                 initializer=identity_initializer())


        conv = tf.nn.atrous_conv2d(batch_input, filter, rate,
                            padding=padding)
        return conv

def deconv(batch_input, out_channels):
    """
    Add transposed convolution to graph.

    [batch, in_height, in_width, in_channels],
    [filter_width, filter_height, out_channels, in_channels]
    => [batch, out_height, out_width, out_channels]
    """
    with tf.variable_scope("deconv"):
        sizes = [int(d) for d in batch_input.get_shape()]
        batch, in_height, in_width, in_channels = sizes
        filter = tf.get_variable("filter", [4, 4, out_channels, in_channels],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())

        conv = tf.nn.conv2d_transpose(batch_input, filter,
                                      [batch, in_height * 2, in_width * 2,
                                       out_channels],
                                      [1, 2, 2, 1], padding="SAME")
        return conv

def upsample(batch_input, out_channels, out_shape=None, stride=1, filter_size=3,
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR):
    """
    Add upsampling followed by a convolutional layer to the graph.

    [batch, in_height, in_width, in_channels],
    [filter_width, filter_height, in_channels, out_channels]
    => [batch, out_height, out_width, out_channels]
    """
    with tf.variable_scope("upsample"):
        _, in_height, in_width, in_channels = batch_input.get_shape().as_list()

        if out_shape == None:
            out_shape = (int(in_height)*2, int(in_width)*2)

        if out_shape != (in_height, in_width):
            upsampled_input = tf.image.resize_images(batch_input,
                                                out_shape,
                                                method=method)
        else:
            upsampled_input = batch_input

        filter = tf.get_variable("filter", [filter_size, filter_size,
                                            in_channels,
                                            out_channels],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())

        conv = tf.nn.conv2d(upsampled_input, filter, [1, stride, stride, 1],
                            padding="SAME")
        return conv

def lrelu(x, a):
    """
    Add leaky-relu activation.

    adding these together creates the leak part and linear part
    then cancels them out by subtracting/adding an absolute value term
    leak: a*x/2 - a*abs(x)/2
    linear: x/2 + abs(x)/2
    """
    with tf.name_scope("lrelu"):
        # this block looks like it has 2 inputs
        # on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)

def batchnorm(input):
    """Add bachnorm to layer."""
    with tf.variable_scope("batchnorm"):
        # this block looks like it has 3 inputs on the graph unless we do this
        input = tf.identity(input)
        channels = input.get_shape()[3]
        offset = tf.get_variable("offset", [channels],
                                 dtype=tf.float32,
                                 initializer=init_zeros)
        scale = tf.get_variable("scale", [channels],
                                dtype=tf.float32,
                                initializer=tf.random_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
        epsilon = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset,
                                               scale,
                                               variance_epsilon=epsilon)
        return normalized

def instance_norm(input):
    """based conditional_instance_norm from https://github.com/tensorflow/magenta/blob/master/magenta/models/image_stylization/ops.py."""
    with tf.variable_scope("instancenorm") as s:
        # this block looks like it has 3 inputs on the graph unless we do this
        input = tf.identity(input)
        channels = input.get_shape()[3]
        offset = tf.get_variable("offset", [channels],
                                 dtype=tf.float32,
                                 initializer=init_zeros)
        scale = tf.get_variable("scale", [channels],
                                dtype=tf.float32,
                                initializer=tf.random_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(input, axes=[1, 2], keep_dims=True)

        epsilon = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset,
                                               scale,
                                               variance_epsilon=epsilon)
        return normalized

def adaptive_instance_norm(input, target):
    with tf.variable_scope("ada_instancenorm"):
        # this block looks like it has 3 inputs on the graph unless we do this
        input = tf.identity(input)
        mean, variance = tf.nn.moments(input, axes=[1, 2], keep_dims=True)
        target_mean, target_variance = tf.nn.moments(target, axes=[1, 2], keep_dims=True)
        epsilon = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, 
                                               target_mean,
                                               target_variance,
                                               variance_epsilon=epsilon)

        return normalized

def encoder(encoder_inputs, input_layer_spec, layer_specs, instancenorm=False):
    """Create image encoder. Based on layer specs."""
    layers = []
    named_layers = {}

    named_layers['input'] = encoder_inputs

    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, input_layer_spec]
    with tf.variable_scope("encoder_1"):
        output = conv(encoder_inputs, input_layer_spec, stride=2)
        output = tf.nn.elu(output, 0.2)
        layers.append(output)
        named_layers["encoder_1"] = layers[-1]

    norm = instance_norm if instancenorm else batchnorm
    for out_channels in layer_specs:
        scope_name = "encoder_%d" % (len(layers) + 1)
        with tf.variable_scope(scope_name):
            # [batch, in_height, in_width, in_channels]
            # => [batch, in_height/2, in_width/2, out_channels]
            convolved = conv(layers[-1], out_channels, stride=2)
            output = norm(convolved)
            output = tf.nn.elu(output)
            layers.append(output)
            named_layers[scope_name] = layers[-1]

    _, out_height, out_width, out_channels = layers[-1].get_shape().as_list()
    img_embed = tf.reshape(layers[-1], [-1, out_height*out_width*out_channels])
    return named_layers, img_embed

def decoder(input_layers, layer_specs, output_layer_specs,
        instancenorm=False, upsample_method=None, input_layer_name=None):
    """Create decoder network based on some layerspec."""

    layers = []
    named_layers = {}

    norm = instance_norm if instancenorm else batchnorm
    num_encoder_layers = len(input_layers)
    for decoder_layer, (dropout, skip_layer) in enumerate(layer_specs):
        scope_name = "decoder_%d" % (len(layer_specs) + 1 - decoder_layer) 

        if isinstance(skip_layer, tuple):
            out_height, out_width, out_channels = skip_layer
        else:
            # Number of out channels is equal to the number of channels of
            # the skip connection we'll be concat with
            _, out_height, out_width, out_channels = input_layers[skip_layer].get_shape().as_list()

        with tf.variable_scope(scope_name):
            if decoder_layer == 0:
                input = input_layers[input_layer_name]
            else:
                input = layers[-1]

            # [batch, in_height, in_width, in_channels]
            # => [batch, in_height*2, in_width*2, out_channels]
            if upsample_method:
                output = upsample(input, out_channels,
                                  out_shape=(out_height, out_width),
                                  method=upsample_method)
            else:
                output = deconv(input, out_channels)
            output = norm(output)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            if not isinstance(skip_layer, tuple):
                output = concatenate(values=[output, input_layers[skip_layer]], axis=3)

            output = tf.nn.elu(output)

            layers.append(output)
            named_layers[scope_name] = layers[-1]

    # decoder_1: [batch, 128, 128, ngf * 2]
    # => [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        input = layers[-1]
        out_channels, dropout, skip_layer = output_layer_specs

        if isinstance(skip_layer, tuple):
            out_height, out_width = skip_layer
        else:
            _, out_height, out_width, _ = input_layers[skip_layer].get_shape().as_list()

        if upsample_method:
            output = upsample(input, out_channels,
                              out_shape=(out_height, out_width),
                              method=upsample_method)
        else:
            output = deconv(input, out_channels)

        if dropout > 0.0:
            output = tf.nn.dropout(output, keep_prob=1 - dropout)

        if not isinstance(skip_layer, tuple):
            output = concatenate(values=[output, input_layers[skip_layer]], axis=3)

        output = tf.tanh(output)

        layers.append(output)
        named_layers["decoder_1"] = layers[-1]

    return layers[-1]


def discriminator(discrim_inputs, discrim_targets, ndf, instancenorm=False):
    """Create discriminator network. PatchGAN, set discrim_inputs to None for unconditional"""
    n_layers = 3
    layers = []

    # 2x [batch, height, width, in_channels]
    # => [batch, height, width, in_channels * 2]
    if discrim_inputs != None:
        input = concatenate(values=[discrim_inputs, discrim_targets], axis=3)
    else:
        input = discrim_targets

    # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
    with tf.variable_scope("layer_1"):
        convolved = conv(input, ndf, stride=2)
        rectified = lrelu(convolved, 0.2)
        layers.append(rectified)

    # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
    # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
    # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
    norm = instance_norm if instancenorm else batchnorm
    for i in range(n_layers):
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            out_channels = ndf * min(2**(i+1), 8)
            # last layer here has stride 1
            stride = 1 if i == n_layers - 1 else 2
            convolved = conv(layers[-1], out_channels, stride=stride)
            normalized = norm(convolved)
            rectified = lrelu(normalized, 0.2)
            layers.append(rectified)

    # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
    with tf.variable_scope("layer_%d" % (len(layers) + 1)):
        convolved = conv(rectified, out_channels=1, stride=1)
        output = tf.sigmoid(convolved)
        layers.append(output)

    return layers[-1]
