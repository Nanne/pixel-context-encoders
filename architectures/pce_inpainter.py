"""Couple of layers implemented in tensorflow."""
import tensorflow as tf
import ops

FLAGS = tf.app.flags.FLAGS  # parse config

def encoder(encoder_inputs, ngf, number_layers=4, downsample=2):
    """Create image encoder based on layer specs."""

    layers = []
    layers.append(encoder_inputs)

    for n in range(downsample):
        scope_name = "encoder_%d" % len(layers)
        with tf.variable_scope(scope_name):
            convolved = ops.conv(layers[-1], ngf, stride=2)
            output = ops.batchnorm(convolved)
            output = tf.nn.elu(output)
            layers.append(output)

    for n in range(number_layers):
        scope_name = "encoder_%d" % len(layers)
        with tf.variable_scope(scope_name):
            dilation = 2 ** (len(layers) - 2)
            convolved = ops.dilated_conv(layers[-1], ngf, rate=dilation)
            output = ops.batchnorm(convolved)
            output = tf.nn.elu(output)
            layers.append(output)

    return layers, None

def decoder(input_layers, ngf, generator_output_channels,
        drop_prob=0.5, masks=None, number_layers=1, upsample=2):
    """Create decoder network based on layerspec."""

    layers = []

    num_encoder_layers = len(input_layers)
    for decoder_layer in range(number_layers):
        scope_name = "decoder_%d" % (number_layers + 2 - len(layers)) 

        with tf.variable_scope(scope_name):
            if decoder_layer == 0:
                input = input_layers[-1]
            else:
                input = layers[-1]

            output = ops.conv(input, ngf, stride=1, filter_size=3)
            output = ops.batchnorm(output)
            output = tf.nn.elu(output)
            layers.append(output)

    for up_layer in range(upsample-1): 
        scope_name = "decoder_%d" % (number_layers + 2 - len(layers)) 
        with tf.variable_scope(scope_name):
            input = layers[-1]
            new_size = tf.gather(tf.shape(input), [1,2]) * 2
            output = ops.upsample(input, ngf, out_shape=new_size)
            output = ops.batchnorm(output)
            output = tf.nn.elu(output)
            layers.append(output)
        
    with tf.variable_scope("decoder_1"):
        input = layers[-1]
        new_size = tf.gather(tf.shape(input), [1,2]) * 2
        output = ops.upsample(input, generator_output_channels, out_shape=new_size)
        output = tf.tanh(output)

        layers.append(output)

    outputs = tf.multiply(layers[-1], masks)
    outputs = tf.add(outputs, input_layers[0])

    return outputs

def discriminator(discrim_targets, ndf, instancenorm=False):
    return ops.discriminator(None, discrim_targets, ndf, instancenorm=False)

