from util import to_namedtuple
import architectures.ops as ops
import tensorflow as tf

EPS = 1e-12
FLAGS = tf.app.flags.FLAGS  # parse config

if FLAGS.architecture == "PCE":
    from architectures.pce_inpainter import encoder, decoder, discriminator
else:
    raise ValueError('Unknown architecture option')

def create_model(e):
    """Create network."""
    m = {}
    with tf.variable_scope("generator") as scope:
        # Image encoder
        with tf.name_scope("encoder"):
            encoder_activations, image_embedding = encoder(e.inputs,
                                                           FLAGS.ngf)
        # Image decoder
        if e.targets != None:
            with tf.name_scope("decoder"):
                out_channels = int(e.targets.get_shape()[-1])

                outputs = decoder(encoder_activations, FLAGS.ngf,
                              out_channels, drop_prob=0.5, masks=e.masks)

                m['outputs'] = outputs

    # Add discriminator and GAN loss
    # create two copies of discriminator,
    # one for real pairs and one for fake pairs
    # they share the same underlying variables
    with tf.name_scope("real_discriminator"):
        with tf.variable_scope("discriminator"):
            # 2x [batch, height, width, channels]
            # => [batch, 30, 30, 1]
            predict_real = discriminator(e.targets, FLAGS.ndf)

    with tf.name_scope("fake_discriminator"):
        with tf.variable_scope("discriminator", reuse=True):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_fake = discriminator(outputs, FLAGS.ndf)

    with tf.name_scope("discriminator_loss"):
        # minimizing -tf.log will try to get inputs to 1
        # predict_real => 1
        # predict_fake => 0
        discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))
    m['predict_real'] = predict_real
    m['predict_fake'] = predict_fake
    m['discriminator_loss'] = discrim_loss

    with tf.name_scope("generator_loss"):
        # predict_fake => 1
        # abs(targets - outputs) => 0
        # Compute softmax on target and output image
        # Compute Bhataccarya distance
        gen_loss = 0
        gen_loss_content = 0
        if e.targets != None:
            if outputs.get_shape() == e.masks.get_shape():
                batch_n = outputs.get_shape().as_list()[0]

                bool_mask = tf.equal(e.masks, 1)
                _, width, height, n_feat_map = bool_mask.get_shape().as_list()
                bool_mask = tf.reshape( bool_mask, [-1, width*height*n_feat_map] )

                m_outputs = tf.reshape( outputs, [-1, width*height*n_feat_map] )
                m_outputs = tf.boolean_mask(m_outputs, bool_mask)
                m_outputs = tf.reshape( m_outputs, [batch_n, -1] )

                m_targets = tf.reshape( e.targets, [-1, width*height*n_feat_map] )
                m_targets = tf.boolean_mask(m_targets, bool_mask)
                m_targets = tf.reshape( m_targets, [batch_n, -1] )
            else:
                m_outputs = outputs
                m_targets = e.targets

            if FLAGS.bat_loss > 0:
                logits_pred = tf.reshape(m_outputs, [FLAGS.batch_size, -1])
                target_flat = tf.reshape(m_targets, [FLAGS.batch_size, -1])
                prob_pred = tf.nn.softmax(logits_pred)
                prob_target = tf.nn.softmax(target_flat)
                gen_loss_bat = -tf.log(tf.reduce_sum(tf.sqrt(tf.multiply(prob_pred,
                                                             prob_target))))
                gen_loss_content += FLAGS.bat_loss * gen_loss_bat

            # this weight stuff is written a bit hacky, clean up at some point
            #TODO make work for invert_mask
            if FLAGS.overlap > 0 and FLAGS.overlap_weight and not FLAGS.invert_mask and not FLAGS.architecture == "PCE":
                w, h, c = m_outputs.get_shape().as_list()[1:]
                overlap = FLAGS.overlap
                overlap_weight = tf.ones([w-overlap*2,h-overlap*2,c])
                overlap_weight = tf.pad(overlap_weight, [[overlap, overlap], [overlap, overlap], [0,0]])
                overlap_weight = (overlap_weight * -9) + 10 # 10 times the weight on overlap
            else:
                overlap_weight = tf.ones_like(m_outputs)

            if FLAGS.tv_loss > 0:
                recon = e.reconstruct_inpaint(e.images, outputs, e.masks)
                gen_loss_TV = tf.reduce_sum(tf.image.total_variation(recon))
                gen_loss_content += FLAGS.tv_loss * gen_loss_TV

            if FLAGS.l1_loss > 0:
                l1_error = tf.abs(m_targets - m_outputs) * overlap_weight
                gen_loss_L1 = tf.reduce_mean(l1_error)
                gen_loss_content += FLAGS.l1_loss * gen_loss_L1
            if FLAGS.l2_loss > 0:
                l2_error = tf.pow(m_targets - m_outputs,2) * overlap_weight
                gen_loss_L2 = tf.reduce_mean(l2_error)
                gen_loss_content += FLAGS.l2_loss * gen_loss_L2

            gen_loss += FLAGS.content_weight * gen_loss_content
        with tf.name_scope("generator_gan_loss"):
            gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
            gen_loss += gen_loss_GAN * FLAGS.gan_weight

    with tf.name_scope("discriminator_train"):
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        discrim_optim = tf.train.AdamOptimizer(FLAGS.lr, FLAGS.beta1)
        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)
        m['discrim_grads_and_vars'] = discrim_grads_and_vars

    with tf.name_scope("generator_train"):
        with tf.control_dependencies([discrim_train]):
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            gen_optim = tf.train.AdamOptimizer(FLAGS.lr, FLAGS.beta1)
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)
            m['gen_grads_and_vars'] = gen_grads_and_vars

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    losses = []
    if e.targets != None:
        losses.append(gen_loss_content)
    losses.append(discrim_loss)
    losses.append(gen_loss_GAN)
    update_losses = ema.apply(losses)
    if e.targets != None:
        m['gen_loss_content'] = ema.average(gen_loss_content)
    m['discrim_loss'] = ema.average(discrim_loss)
    m['gen_loss_GAN'] = ema.average(gen_loss_GAN)

    global_step = tf.contrib.framework.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)
    m['train'] = tf.group(update_losses, incr_global_step, gen_train)
    model = to_namedtuple(m, "Model")
    return model
