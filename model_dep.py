import tensorflow as tf
import time
from cell import ConvGRUCell
from ConvGRU import ConvGRUCell as ConvGRUCell_1
from layers import deconv2d
from layers import lrelu
from layers import linear
from layers import conv2d
from ops import gdl
import numpy as np
#from ops import *
#from utils import *

LAMBDA = 10 # Gradient penalty lambda hyperparameter
lamda = 200 # wights of the function 8 in wgan-gp
alpha_loss = 1
beta_loss = 0.02

batch_size = 32
timesteps = 100
shape = [640, 480]
kernel = [5, 5]
channels = 3
filters = 12
iters = 0
num_iters = 200000

def add_deconv(input_1, input_2, input_3, input_4):

    with tf.variable_scope('fina_layer'):

        outputs_4 = input_4

        output_shape_3=[input_4.shape[0], input_4.shape[1]*2, input_4.shape[2]*2, input_4.shape[3]]
        outputs_3 = lrelu(input_4)
        outputs_3 = deconv2d(outputs_3, output_shape_3)
        outputs_3 = lrelu(outputs_3)
        res_3 = conv2d(input_3, 64, k_h=3, k_w=3, d_h=1, d_w=1)
        res_3 = tf.nn.relu(res_3)
        res_3 = conv2d(res_3, output_shape_3[-1], k_h=3, k_w=3, d_h=1, d_w=1)
        outputs_3 = tf.add(outputs_3, tf.nn.relu(res_3))
        outputs_3 = lrelu(outputs_3)

        output_shape_2=[input_3.shape[0], input_3.shape[1]*2, input_3.shape[2]*2, input_3.shape[3]]
        outputs_2 = lrelu(input_3)
        outputs_2 = deconv2d(outputs_2, output_shape_2)
        outputs_2 = lrelu(outputs_2)
        res_2 = conv2d(input_2, 64, k_h=3, k_w=3, d_h=1, d_w=1)
        res_2 = tf.nn.relu(res_2)
        res_2 = conv2d(res_2, output_shape_2[-1], k_h=3, k_w=3, d_h=1, d_w=1)
        outputs_2 = tf.add(outputs_2, tf.nn.relu(res_2))
        outputs_2 = lrelu(outputs_2)

        output_shape_1=[input_2.shape[0], input_2.shape[1]*2, input_2.shape[2]*2, input_2.shape[3]]
        outputs_1 = lrelu(input_2)
        outputs_1 = deconv2d(outputs_1, output_shape_1)
        outputs_1 = lrelu(outputs_1)
        res_1 = conv2d(input_1, 64, k_h=3, k_w=3, d_h=1, d_w=1)
        res_1 = tf.nn.relu(res_1)
        res_1 = conv2d(res_1, output_shape_2[-1], k_h=3, k_w=3, d_h=1, d_w=1)
        outputs_1 = tf.add(outputs_1, tf.nn.relu(res_1))
        outputs_1 = lrelu(outputs_1)

        return outputs_1, outputs_2, outputs_3, outputs_4

def gen(inputs, state_init=None):

    inputs_1, inputs_2, inputs_3, inputs_4 = tf.split(inputs, 4) # here 4 represents 4 layers,

    with tf.variable_scope(tf.get_variable_scope()):

        with tf.variable_scope('gen_enc_1'):
            cell_1 = ConvGRUCell_1(shape[1], filters[1], kernel[1], initializer=tf.truncated_normal_initializer(stddev=0.01))
            if state_init is None:
                init = cell_1.zero_state(inputs_1.shape[1].value, tf.float32)
            else:
                init = state_init[0]
            output_1, states_1 = tf.nn.dynamic_rnn(cell_1, inputs_1,
                                                   initial_state=init,
                                                   dtype=tf.float32, time_major=True)
            output_final_1 = []
            for i in xrange(output_1.shape[0].value):
                output_1_temp = tf.nn.max_pool(output_1[i], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                output_final_1.append(output_1_temp)
            output_final_1 = tf.stack(output_final_1)

        with tf.variable_scope('gen_enc_2'):
            cell_2 = ConvGRUCell(shape[2], filters[2], kernel[2], initializer=tf.truncated_normal_initializer(stddev=0.01))
            if state_init is None:
                init_2 = cell_2.zero_state(inputs_2.shape[1].value, tf.float32)
            else:
                init_2 = state_init[1]
            output_2, states_2 = tf.nn.dynamic_rnn(cell_2, tf.concat(inputs_2, output_final_1, 2),
                                                   initial_state=init_2, dtype=tf.float32,
                                                   time_major=True)
            output_final_2 = []
            for i in xrange(output_2.shape[0].value):
                output_2_temp = tf.nn.max_pool(output_2[i], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                output_final_1.append(output_2_temp)
            output_final_2 = tf.stack(output_final_2)

        with tf.variable_scope('gen_enc_3'):
            cell_3 = ConvGRUCell(shape[3], filters[3], kernel[3], initializer=tf.truncated_normal_initializer(stddev=0.01))
            if state_init is None:
                init_3 = cell_3.zero_state(inputs_3.shape[1].value, tf.float32)
            else:
                init_3 = state_init[2]
            output_3, states_3 = tf.nn.dynamic_rnn(cell_3, tf.concat(inputs_3, output_final_2, 2),
                                                   initial_state=init_3, dtype=tf.float32,
                                                   time_major=True)
            output_final_3 = []
            for i in xrange(output_3.shape[0].value):
                output_3_temp = tf.nn.max_pool(output_3[i], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                output_final_3.append(output_3_temp)
            output_final_3 = tf.stack(output_final_3)

        with tf.variable_scope('gen_enc_4'):
            cell_4 = ConvGRUCell(shape[4], filters[4], kernel[4], initializer=tf.truncated_normal_initializer(stddev=0.01))
            if state_init is None:
                init_4 = cell_4.zero_state(inputs_4.shape[1].value, tf.float32)
            else:
                init_4 = state_init[3]
            _, states_4 = tf.nn.dynamic_rnn(cell_4, tf.concat(inputs_4, output_final_3, 2),
                                            initial_state=init_4,
                                            dtype=tf.float32, time_major=True)

        final_output_1 = []
        final_output_2 = []
        final_output_3 = []
        final_output_4 = []
        final_state = []
        # interm_out_1 = []
        # interm_out_2 = []
        # interm_out_3 = []
        # interm_out_4 = []
        # dec_sta_1 = []
        # dec_sta_2 = []
        # dec_sta_3 = []
        # dec_sta_4 = []

        for step in xrange(len(inputs)):

                if step == 0:

                    with tf.variable_scope('gen_dec_1'):
                         cell_dec_1 = ConvGRUCell_1(shape[1], filters[1], kernel[1])
                         dec_output_1, dec_states_1 = tf.nn.dynamic_rnn(cell_dec_1, tf.zeros_like(), initial_state=states_1,
                                                           dtype=tf.float32, time_major=True)
                         dec_output_final_1 = []
                         for i in xrange(dec_output_1.shape[0].value):
                             dec_output_1_temp = tf.nn.max_pool(dec_output_1[i], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                   padding='SAME')
                             dec_output_final_1.append(dec_output_1_temp)
                         dec_output_final_1 = tf.stack(dec_output_final_1)

                    with tf.variable_scope('gen_dec_2'):
                         cell_dec_2 = ConvGRUCell(shape[2], filters[2], kernel[2])
                         dec_output_2, dec_states_2 = tf.nn.dynamic_rnn(cell_dec_2, tf.concat(tf.zeros_like(), dec_output_final_1,2), initial_state=states_2,
                                                           dtype=tf.float32, time_major=True)
                         dec_output_final_2 = []
                         for i in xrange(dec_output_2.shape[0].value):
                             dec_output_2_temp = tf.nn.max_pool(dec_output_2[i], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                   padding='SAME')
                             dec_output_final_2.append(dec_output_2_temp)
                         dec_output_final_2 = tf.stack(dec_output_final_2)

                    with tf.variable_scope('gen_dec_3'):
                         cell_dec_3 = ConvGRUCell(shape[3], filters[3], kernel[3])
                         dec_output_3, dec_states_3 = tf.nn.dynamic_rnn(cell_dec_3, tf.concat(tf.zeros_like(),dec_output_final_2,2), initial_state=states_3,
                                                           dtype=tf.float32, time_major=True)
                         dec_output_final_3 = []
                         for i in xrange(dec_output_3.shape[0].value):
                             dec_output_3_temp = tf.nn.max_pool(dec_output_3[i], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                   padding='SAME')
                             dec_output_final_3.append(dec_output_3_temp)
                         dec_output_final_3 = tf.stack(dec_output_final_3)

                    with tf.variable_scope('gen_dec_4'):
                         cell_dec_4 = ConvGRUCell(shape[4], filters[4], kernel[4])
                         dec_output_4, dec_states_4 = tf.nn.dynamic_rnn(cell_dec_4, tf.concat(tf.zeros_like(),dec_output_final_3,2), initial_state=states_4,
                                                                      dtype=tf.float32, time_major=True)

                    with tf.variable_scope('gen_dec_logits'):
                         interm_1, interm_2, interm_3, interm_4 = add_deconv(dec_states_1, dec_states_2, dec_states_3, dec_states_4)

                    interm_out_1 = []
                    interm_out_2 = []
                    interm_out_3 = []
                    interm_out_4 = []
                    interm_out_1.append(interm_1)
                    final_output_1.append(interm_out_1)
                    interm_out_1 = tf.stack(interm_out_1)
                    interm_out_2.append(interm_2)
                    final_output_2.append(interm_out_2)
                    interm_out_2 = tf.stack(interm_out_2)
                    interm_out_3.append(interm_3)
                    final_output_3.append(interm_out_3)
                    interm_out_3 = tf.stack(interm_out_3)
                    interm_out_4.append(interm_4)
                    final_output_4.append(interm_out_4)
                    interm_out_4 = tf.stack(interm_out_4)

                else:

                    with tf.variable_scope('gen_dec_1', reuse=True):
                         cell_dec_1 = ConvGRUCell_1(shape[1], filters[1], kernel[1])
                         dec_output_1, dec_states_1 = tf.nn.dynamic_rnn(cell_dec_1, interm_out_1, initial_state=dec_states_1,
                                                           dtype=tf.float32, time_major=True)
                         dec_output_final_1 = []
                         for i in xrange(dec_output_1.shape[0].value):
                             dec_output_1_temp = tf.nn.max_pool(dec_output_1[i], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                   padding='SAME')
                             dec_output_final_1.append(dec_output_1_temp)
                         dec_output_final_1 = tf.stack(dec_output_final_1)

                    with tf.variable_scope('gen_dec_2', reuse=True):
                         cell_dec_2 = ConvGRUCell(shape[2], filters[2], kernel[2])
                         dec_output_2, dec_states_2 = tf.nn.dynamic_rnn(cell_dec_2, tf.concat(interm_out_2,dec_output_final_1,2), initial_state=dec_states_2,
                                                           dtype=tf.float32, time_major=True)
                         dec_output_final_2 = []
                         for i in xrange(dec_output_2.shape[0].value):
                             dec_output_2_temp = tf.nn.max_pool(dec_output_2[i], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                   padding='SAME')
                             dec_output_final_2.append(dec_output_2_temp)
                         dec_output_final_2 = tf.stack(dec_output_final_2)

                    with tf.variable_scope('gen_dec_3', reuse=True):
                         cell_dec_3 = ConvGRUCell(shape[3], filters[3], kernel[3])
                         dec_output_3, dec_states_3 = tf.nn.dynamic_rnn(cell_dec_3, tf.concat(interm_out_3,dec_output_final_2,2), initial_state=dec_states_3,
                                                           dtype=tf.float32, time_major=True)
                         dec_output_final_3 = []
                         for i in xrange(dec_output_3.shape[0].value):
                             dec_output_3_temp = tf.nn.max_pool(dec_output_3[i], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                   padding='SAME')
                             dec_output_final_3.append(dec_output_3_temp)
                         dec_output_final_3 = tf.stack(dec_output_final_3)

                    with tf.variable_scope('gen_dec_4', reuse=True):
                         cell_dec_4 = ConvGRUCell(shape[4], filters[4], kernel[4])
                         dec_output_4, dec_states_4 = tf.nn.dynamic_rnn(cell_dec_4, tf.concat(interm_out_4,dec_output_final_3,2), initial_state=dec_states_4,
                                                                        dtype=tf.float32, time_major=True)

                    with tf.variable_scope('gen_dec_logits', reuse=True):
                         interm_1, interm_2, interm_3, interm_4 = add_deconv(dec_states_1, dec_states_2, dec_states_3, dec_states_4)

                    interm_out_1 = []
                    interm_out_2 = []
                    interm_out_3 = []
                    interm_out_4 = []
                    interm_out_1.append(interm_1)
                    final_output_1.append(interm_out_1)
                    interm_out_1 = tf.stack(interm_out_1)
                    interm_out_2.append(interm_2)
                    final_output_2.append(interm_out_2)
                    interm_out_2 = tf.stack(interm_out_2)
                    interm_out_3.append(interm_3)
                    final_output_3.append(interm_out_3)
                    interm_out_3 = tf.stack(interm_out_3)
                    interm_out_4.append(interm_4)
                    final_output_4.append(interm_out_4)
                    interm_out_4 = tf.stack(interm_out_4)

                    if step == len(inputs):
                         final_state.append(dec_states_1)
                         final_state.append(dec_states_2)
                         final_state.append(dec_states_3)
                         final_state.append(dec_states_4)

        final_output_1 = tf.stack(final_output_1) # the final output would be final_output_1
        final_output_2 = tf.stack(final_output_2)
        final_output_3 = tf.stack(final_output_3)
        final_output_4 = tf.stack(final_output_4)
        final_state = tf.stack(final_state)

    return final_output_1, final_output_2, final_output_3, final_output_4, final_state


def disc(inputs):

    #with tf.name_scope('disc'):

        with tf.variable_scope('disc'):

             h0 = lrelu(conv2d(inputs, filters, k_h=5, k_w=5, d_h=2, d_w=2))
            # h0 is (112 x 112 x self.df_dim)
             h1 = lrelu(conv2d(h0, filters*2, k_h=5, k_w=5, d_h=2, d_w=2))
            # h1 is (56 x 56 x self.df_dim*2)
             h2 = lrelu(conv2d(h1, filters*4, k_h=5, k_w=5, d_h=2, d_w=2))
            # h2 is (28x 28 x self.df_dim*4)
             h3 = lrelu(conv2d(h2, filters*8, k_h=5, k_w=5, d_h=2, d_w=2))
            # h3 is (14 x 14 x self.df_dim*8)
             h4 = linear(tf.reshape(h3, [inputs[0], -1]), 1) # replace inputs[0] as self.batch_size

             return h4



while iters < num_iters:

# idx = 0
    counter = iters + 1
    start_time = time.time()
    data = [] ## read file, the first dimension could be batch or time steps;
    gt = []## read file for groundtruth

## here the idx needs to be shffuled, check how!!!

    for idx in xrange(data[0].value):

        x_in = tf.placeholder(tf.float32, [idx, timesteps] + shape + [channels]) ## replace the 'idx' with batch size
        y_gt = tf.placeholder(tf.float32, [idx, timesteps] + shape + [channels])
        x_in = tf.transpose(x_in, (1, 0, 2, 3, 4)) # make sure the input is 5D
        y_gt = tf.transpose(y_gt, (1, 0, 2, 3, 4)) # make sure the gt is 5D

        pred_re = tf.placeholder(tf.float32, [timesteps, idx] + shape + [channels]) ## replace the 'idx' with batch size
        recon_re = tf.placeholder(tf.float32, [timesteps, idx] + shape + [channels])

        data = ## read each batch and make it 5D
        gt =

        with tf.variable_scope('pred_gen'):
            pred = []
            pred_1, pred_2, pred_3, pred_4, pred_state = gen(x_in)
            pred.append(pred_1)
            pred.append(pred_2)
            pred.append(pred_3)
            pred.append(pred_4)
            pred = tf.stack(pred)

        with tf.variable_scope('recon_gen'):
            recon, _, _, _, _ = gen(pred, state_init=pred_state)

        loss_pred_g_sum = 0.
        loss_pred_d_sum = 0.
        loss_pred_sum = 0.
        loss_recon_g_sum = 0.
        loss_recon_d_sum = 0.
        loss_recon_sum = 0.

        for step in xrange(pred_1[0].value):  ## the original dual loss paper, the input are pairs of images, thus two g and d. In my case,

            loss_pred = tf.reduce_mean(tf.abs(pred_1[step] - y_gt[step]))
            with tf.variable_scope('loss_pred'):
                loss_pred_g = beta_loss * tf.reduce_mean(-disc(pred_1[step])) + alpha_loss * (loss_pred + gdl(pred_1[step], y_gt[step], 1))  # loss of G; L1 distance between pred and gt; gdl may be redundant
            with tf.variable_scope('loss_pred', reuse=True):
                loss_pred_d = beta_loss * tf.reduce_mean(disc(y_gt[step]))

            loss_recon = tf.reduce_mean(tf.abs(recon[step] - x_in[step]))
            with tf.variable_scope('loss_recon'):
                loss_recon_g = beta_loss * tf.reduce_mean(-disc(recon[step])) + alpha_loss * (loss_recon + gdl(recon[step], x_in[step], 1))  # loss of G; L1 distance between recon and input; gdl may be redundant
            with tf.variable_scope('loss_recon', reuse=True):
                loss_recon_d = beta_loss * tf.reduce_mean(disc(x_in[step]))

            loss_pred_g_sum += loss_pred_g
            loss_pred_d_sum += loss_pred_d
            loss_pred_sum += loss_pred

            loss_recon_g_sum += loss_recon_g
            loss_recon_d_sum += loss_recon_d
            loss_recon_sum += loss_recon

        loss_pred_g_sum = tf.divide(loss_pred_g_sum, pred_1[0].value)
        sum_pred_g = tf.summary.scalar('cost_pred_g', loss_pred_g_sum)

        loss_pred_d_sum = tf.divide(loss_pred_d_sum, pred_1[0].value)

        loss_pred_sum = tf.divide(loss_pred_sum, pred_1[0].value)
        pred_sum = tf.summary.scalar('cost_pred', loss_pred_sum)
        sum_pred = tf.summary.merge([sum_pred_g, pred_sum])

        loss_recon_g_sum = tf.divide(loss_recon_g_sum, pred_1[0].value)
        sum_recon_g = tf.summary.scalar('cost_recon_g', loss_recon_g_sum)

        loss_recon_d_sum = tf.divide(loss_recon_d_sum, pred_1[0].value)

        loss_recon_sum = tf.divide(loss_recon_sum, pred_1[0].value)
        recon_sum = tf.summary.scalar('cost_recon', loss_recon_sum)
        sum_recon = tf.summary.merge([sum_recon_g, recon_sum])

# WGAN_GP:

        alpha = tf.random_uniform(shape=[pred_1[1].value, 1], minval=0., maxval=1.) ## Here replaing pred_1[1].value with batch_size.
        differences_pred = pred_1 - y_gt ##since here, the extra loss presents the difference between real data and generated ones, make them separated make sense
        differences_recon = recon - x_in
        interpolates_pred = y_gt + (alpha * differences_pred)
        interpolates_recon = x_in + (alpha * differences_recon)

        with tf.variable_scope('loss_pred', reuse=True):
            gradients_pred = tf.gradients(disc(interpolates_pred), [interpolates_pred])[0] ## here the disc has to be specified, hence the diff have to be separated
        with tf.variable_scope('loss_recon', reuse=True):
            gradients_recon = tf.gradients(disc(interpolates_recon), [interpolates_recon])[0]

        slopes_pred = tf.sqrt(tf.reduce_sum(tf.square(gradients_pred), reduction_indices=[1]))
        slopes_recon = tf.sqrt(tf.reduce_sum(tf.square(gradients_recon), reduction_indices=[1]))
        gradient_penalty_pred = tf.reduce_mean((slopes_pred-1.) ** 2)
        gradient_penalty_recon = tf.reduce_mean((slopes_recon-1.) ** 2)

        loss_pred_d_sum += tf.divide(LAMBDA * gradient_penalty_pred, pred_1[0].value)
        sum_pred_d = tf.summary.scalar('cost_pred_d', loss_pred_d_sum)

        loss_recon_d_sum += tf.divide(LAMBDA * gradient_penalty_recon, pred_1[0].value)
        sum_recon_d = tf.summary.scalar('cost_recon_d', loss_recon_d_sum)

## define trainable variables:

        t_vars = tf.trainable_variables()

        gen_pred_vars = [var for var in t_vars if 'pred_gen' in var.name]
        disc_pred_vars = [var for var in t_vars if 'loss_pred' in var.name]
        gen_recon_vars = [var for var in t_vars if 'recon_gen' in var.name]
        disc_recon_vars = [var for var in t_vars if 'loss_recon' in var.name]

        gen_pred_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5, decay=0.9).minimize(loss_pred_g_sum, var_list=gen_pred_vars)
        disc_pred_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5, decay=0.9).minimize(loss_pred_d_sum, var_list=disc_pred_vars)
        gen_recon_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5, decay=0.9).minimize(loss_recon_g_sum, var_list=gen_recon_vars)
        disc_recon_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5, decay=0.9).minimize(loss_recon_d_sum, var_list=disc_recon_vars)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

## forward G:
            pred_gen, recon_gen = sess.run([pred, recon], feed_dict={x_in: data})
            pred_gen_1, pred_gen_2, pred_gen_3, pred_gen_4 = tf.split(pred_gen, 4)

## updating G_pred:
            _, loss_pg, loss_2gt, summary_str = sess.run([gen_pred_train_op, loss_pred_g_sum, loss_pred_sum, sum_pred], feed_dict={x_in: data, y_gt: gt})

## updating D_pred, should be with n_critic
            _, loss_pd, summary_str = sess.run([disc_pred_train_op, loss_pred_d_sum, sum_pred_d], feed_dict={pred_re: pred_gen_1, y_gt: gt}) ## the idx of the input

            _, loss_rg, loss_2in, summary_str = sess.run([gen_recon_train_op, loss_recon_g_sum, loss_recon_sum, sum_recon], feed_dict={pred_re: pred_gen, x_in: data})

            _, loss_rd, summary_str = sess.run([disc_recon_train_op, loss_recon_d_sum, sum_recon_d], feed_dict={recon_gen: recon_gen, x_in: data})




        iters += 1
        counter += 1













