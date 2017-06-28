import tensorflow as tf
from cell import ConvGRUCell
from ConvGRU import ConvGRUCell as ConvGRUCell_1
from layers import deconv2d
from layers import lrelu
from layers import linear
from layers import conv2d
import numpy as np
# from ops import *
# from utils import *

LAMBDA = 10 # Gradient penalty lambda hyperparameter
lamda = 200 # wights of the function 8 in wgan-gp

batch_size = 32
timesteps = 100
shape = [640, 480]
kernel = [3, 3]
channels = 3
filters = 12

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

def gen(input, state_init=None):

    # with tf.variable_scope('inputs'):
    #
    #     xs = tf.placeholder(tf.float32, [self._batch_size, self._time_steps, self._input_size], name='xs')
    #     ys = tf.placeholder(tf.float32, [self._batch_size, self._time_steps, self._output_size], name='ys')

    with tf.variable_scope(tf.get_variable_scope()):

        with tf.variable_scope('gen_enc_1'):
            cell_1 = ConvGRUCell_1(shape[1], filters[1], kernel[1], initializer=tf.truncated_normal_initializer(stddev=0.01))
            if state_init is None:
                init = cell_1.zero_state(input.shape[1].value, tf.float32)
            else:
                init = state_init[0]
            output_1, states_1 = tf.nn.dynamic_rnn(cell_1, input,
                                                   initial_state=init,
                                                   dtype=tf.float32, time_major=True)
            output_final_1 = []
            for i in range(output_1.shape[0].value):
                output_1_temp = tf.nn.max_pool(output_1[i], ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
                output_final_1.append(output_1_temp)
            output_final_1 = tf.stack(output_final_1)

        with tf.variable_scope('gen_enc_2'):
            cell_2 = ConvGRUCell(shape[2], filters[2], kernel[2], initializer=tf.truncated_normal_initializer(stddev=0.01))
            if state_init is None:
                init_2 = cell_2.zero_state(input.shape[1].value, tf.float32)
            else:
                init_2 = state_init[1]
            output_2, states_2 = tf.nn.dynamic_rnn(cell_2, tf.concat(input[2], output_final_1, 2),
                                                   initial_state=init_2, dtype=tf.float32,
                                                   time_major=True)
            output_final_2 = []
            for i in range(output_2.shape[0].value):
                output_2_temp = tf.nn.max_pool(output_2[i], ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
                output_final_1.append(output_2_temp)
            output_final_2 = tf.stack(output_final_2)

        with tf.variable_scope('gen_enc_3'):
            cell_3 = ConvGRUCell(shape[3], filters[3], kernel[3], initializer=tf.truncated_normal_initializer(stddev=0.01))
            if state_init is None:
                init_3 = cell_3.zero_state(input.shape[1].value, tf.float32)
            else:
                init_3 = state_init[2]
            output_3, states_3 = tf.nn.dynamic_rnn(cell_3, tf.concat(input[3], output_final_2, 2),
                                                   initial_state=init_3, dtype=tf.float32,
                                                   time_major=True)
            output_final_3 = []
            for i in range(output_3.shape[0].value):
                output_3_temp = tf.nn.max_pool(output_3[i], ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
                output_final_3.append(output_3_temp)
            output_final_3 = tf.stack(output_final_3)

        with tf.variable_scope('gen_enc_4'):
            cell_4 = ConvGRUCell(shape[4], filters[4], kernel[4], initializer=tf.truncated_normal_initializer(stddev=0.01))
            if state_init is None:
                init_4 = cell_4.zero_state(input.shape[1].value, tf.float32)
            else:
                init_4 = state_init[3]
            _, states_4 = tf.nn.dynamic_rnn(cell_4, tf.concat(input[4], output_final_3, 2),
                                            initial_state=init_4,
                                            dtype=tf.float32, time_major=True)

        final_output_1 = []
        final_output_2 = []
        final_output_3 = []
        final_output_4 = []
        final_state = []
        for step in range(len(input)):

                if step == 0:

                    with tf.variable_scope('gen_dec_1'):
                         cell_dec_1 = ConvGRUCell_1(shape[1], filters[1], kernel[1])
                         dec_output_1, dec_states_1 = tf.nn.dynamic_rnn(cell_dec_1, tf.zeros_like(), initial_state=states_1,
                                                           dtype=tf.float32, time_major=True)
                         dec_output_final_1 = []
                         for i in range(dec_output_1.shape[0].value):
                             dec_output_1_temp = tf.nn.max_pool(dec_output_1[i], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                   padding='SAME')
                             dec_output_final_1.append(dec_output_1_temp)
                         dec_output_final_1 = tf.stack(dec_output_final_1)

                    with tf.variable_scope('gen_dec_2'):
                         cell_dec_2 = ConvGRUCell(shape[2], filters[2], kernel[2])
                         dec_output_2, dec_states_2 = tf.nn.dynamic_rnn(cell_dec_2, tf.concat(tf.zeros_like(), dec_output_final_1,2), initial_state=states_2,
                                                           dtype=tf.float32, time_major=True)
                         dec_output_final_2 = []
                         for i in range(dec_output_2.shape[0].value):
                             dec_output_2_temp = tf.nn.max_pool(dec_output_2[i], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                   padding='SAME')
                             dec_output_final_2.append(dec_output_2_temp)
                         dec_output_final_2 = tf.stack(dec_output_final_2)

                    with tf.variable_scope('gen_dec_3'):
                         cell_dec_3 = ConvGRUCell(shape[3], filters[3], kernel[3])
                         dec_output_3, dec_states_3 = tf.nn.dynamic_rnn(cell_dec_3, tf.concat(tf.zeros_like(),dec_output_final_2,2), initial_state=states_3,
                                                           dtype=tf.float32, time_major=True)
                         dec_output_final_3 = []
                         for i in range(dec_output_3.shape[0].value):
                             dec_output_3_temp = tf.nn.max_pool(dec_output_3[i], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                   padding='SAME')
                             dec_output_final_3.append(dec_output_3_temp)
                         dec_output_final_3 = tf.stack(dec_output_final_3)

                    with tf.variable_scope('gen_dec_4'):
                         cell_dec_4 = ConvGRUCell(shape[4], filters[4], kernel[4])
                         dec_output_4, dec_states_4 = tf.nn.dynamic_rnn(cell_dec_4, tf.concat(tf.zeros_like(),dec_output_final_3,2), initial_state=states_4,
                                                                      dtype=tf.float32, time_major=True)

                    with tf.variable_scope('gen_dec_logits'):
                         interm_out_1 = []
                         interm_out_2 = []
                         interm_out_3 = []
                         interm_out_4 = []
                         interm_1, interm_2, interm_3, interm_4 = add_deconv(dec_states_1, dec_states_2, dec_states_3, dec_states_4)
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
                         for i in range(dec_output_1.shape[0].value):
                             dec_output_1_temp = tf.nn.max_pool(dec_output_1[i], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                   padding='SAME')
                             dec_output_final_1.append(dec_output_1_temp)
                         dec_output_final_1 = tf.stack(dec_output_final_1)

                    with tf.variable_scope('gen_dec_2', reuse=True):
                         cell_dec_2 = ConvGRUCell(shape[2], filters[2], kernel[2])
                         dec_output_2, dec_states_2 = tf.nn.dynamic_rnn(cell_dec_2, tf.concat(interm_out_2,dec_output_final_1,2), initial_state=dec_states_2,
                                                           dtype=tf.float32, time_major=True)
                         dec_output_final_2 = []
                         for i in range(dec_output_2.shape[0].value):
                             dec_output_2_temp = tf.nn.max_pool(dec_output_2[i], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                   padding='SAME')
                             dec_output_final_2.append(dec_output_2_temp)
                         dec_output_final_2 = tf.stack(dec_output_final_2)

                    with tf.variable_scope('gen_dec_3', reuse=True):
                         cell_dec_3 = ConvGRUCell(shape[3], filters[3], kernel[3])
                         dec_output_3, dec_states_3 = tf.nn.dynamic_rnn(cell_dec_3, tf.concat(interm_out_3,dec_output_final_2,2), initial_state=dec_states_3,
                                                           dtype=tf.float32, time_major=True)
                         dec_output_final_3 = []
                         for i in range(dec_output_3.shape[0].value):
                             dec_output_3_temp = tf.nn.max_pool(dec_output_3[i], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                   padding='SAME')
                             dec_output_final_3.append(dec_output_3_temp)
                         dec_output_final_3 = tf.stack(dec_output_final_3)

                    with tf.variable_scope('gen_dec_4', reuse=True):
                         cell_dec_4 = ConvGRUCell(shape[4], filters[4], kernel[4])
                         dec_output_4, dec_states_4 = tf.nn.dynamic_rnn(cell_dec_4, tf.concat(interm_out_4,dec_output_final_3,2), initial_state=dec_states_4,
                                                                        dtype=tf.float32, time_major=True)

                    with tf.variable_scope('gen_dec_logits', reuse=True):
                         interm_out_1 = []
                         interm_out_2 = []
                         interm_out_3 = []
                         interm_out_4 = []
                         interm_1, interm_2, interm_3, interm_4 = add_deconv(dec_states_1, dec_states_2, dec_states_3, dec_states_4)
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

                    if step == len(input):
                         final_state.append(dec_states_1)
                         final_state.append(dec_states_1)
                         final_state.append(dec_states_1)
                         final_state.append(dec_states_1)

        final_output_1 = tf.stack(final_output_1)
        final_output_2 = tf.stack(final_output_2)
        final_output_3 = tf.stack(final_output_3)
        final_output_4 = tf.stack(final_output_4)
        final_state = tf.stack(final_state)

    return final_output_1, final_output_2, final_output_3, final_output_4, final_state


def disc(inputs):

    #with tf.name_scope('disc'):

        with tf.variable_scope('disc'):

             h0 = lrelu(conv2d(image, self.df_dim, name=prefix+'d_h0_conv'))
            # h0 is (128 x 128 x self.df_dim)
             h1 = lrelu(conv2d(h0, self.df_dim*2, name=prefix+'d_h1_conv'), name = prefix+'d_bn1')
            # h1 is (64 x 64 x self.df_dim*2)
             h2 = lrelu(conv2d(h1, self.df_dim*4, name=prefix+'d_h2_conv'), name = prefix+ 'd_bn2')
            # h2 is (32x 32 x self.df_dim*4)
             h3 = lrelu(conv2d(h2, self.df_dim*8, d_h=1, d_w=1, name=prefix+'d_h3_conv'), name = prefix+ 'd_bn3')
            # h3 is (16 x 16 x self.df_dim*8)
             h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, prefix+'d_h3_lin')

             return h4

with tf.variable_scope('pred'):
    pred_1, pred_2, pred_3, pred_4, pred_state = gen(inputs)

with tf.variable_scope('recon'):
    recon, _, _, _, _ = gen(pred, state_init=pred_state)
#
# def pred_gen(seq):
#
#     with tf.variable_scope('pred_gen'):
#
#         return gen(seq)
#
# def pred_disc(dec_outputs:
#
#     with tf.name_scope('pred_disc'):
#
#     #with tf.variable_scope('p_disc'):
#
#         return disc(input, reuse=None)
#
# # loss
# hidd_pred, gen_pred = pred_gen(feature_maps)
# disc_pred = pred_disc(gen_pred)
#
# def recon_gen(, state_init=None):
#
#     with tf.name_scope('recon_gen'):
#
#     #with tf.variable_scope('re_gen'):
#
#         return gen(seq, reuse=None)
#
# def recon_disc(, reuse = False):
#
#     with tf.name_scope('recon_disc'):
#
#     #with tf.variable_scope('re_disc'):
#
#         return disc(input, reuse=None)
#
# _, gen_recon = recon_gen(dec_outputs, hidden_pred) # remember here is hidden_pred_reversed
# disc_recon = recon_disc(gen_recon)

loss_pre = tf.reduce_mean(-disc(pred_1) ) + tf.reduce_mean(disc(gt) + lamda * tf.reduce_mean(tf.abs(recon - inputs))# FW_G_output))
loss_disc = tf.reduce_mean(-recon_disc(gen_recon)) + tf.reduce_mean(recon_disc(inputs, reuse=True))

## define trainable variables
t_vars = tf.trainable_variables()

gen_pred_vars = [var for var in t_vars if 'gen_pred_' in var.name]
disc_pred_vars = [var for var in t_vars if 'disc_pred_' in var.name]

gen_recon_vars = [var for var in t_vars if 'gen_recon__' in var.name]
disc_recon_vars = [var for var in t_vars if 'disc_recon_' in var.name]

d_vars = disc_pred_vars + disc_recon_vars

g_vars = gen_pred_vars + gen_recon_vars


    # Gradient penalty
alpha = tf.random_uniform(shape=[BATCH_SIZE,1], minval=0.,maxval=1.)
differences = (gen_pred[] - gt[]) + (gen_recon[] - input) # the index of sampled for diff has to be paired
interpolates = inputs[] + gt + (alpha*differences)
gradients = tf.gradients(disc(interpolates), [interpolates])[0]
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
gradient_penalty = tf.reduce_mean((slopes-1.)**2)
loss += LAMBDA*gradient_penalty

# training
gen_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(loss_pre, var_list = g_vars)
disc_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(loss_disc, var_list = d_vars)

# def add_layer(input, state_init=None):
#
#     #with tf.variable_scope('add_layer'):
#
#         cell = ConvGRUCell(shape, filters, kernel, reuse=True)
#
#         if state_init == None:
#             state_init = cell.zero_state()
#         else:
#             state_init = state_init
#
#         cell_output, cell_state = tf.nn.dynamic_rnn(cell, input, initial_state = state_init, dtype=tf.float32, time_major=True)
#
#         return cell_output, cell_state

        # enc_cell = ConvGRUCell(shape[nl], filters[nl], kernel[nl])
        #
        # if state_init == None:
        #
        #     state_init = enc_cell.zero_state()
        #
        # else:
        #
        #     state_init = state_init
        #
        # if nl > 0:
        #
        #     enc_inputs = tf.concat(feature_maps[nl], enc_outputs[nl - 1], 1)
        # # enc_cell = ConvGRUCell(shape[nl], filters[nl], kernel[nl])
        #     enc_outputs, enc_states = tf.nn.dynamic_rnn(enc_cell, enc_inputs[nl], initial_state = state_init, dtype=tf.float32, time_major=True)
        # else:
        #
        #     enc_inputs = feature_maps[nl]
        # # enc_cell = ConvGRUCell(shape[nl], filters[nl], kernel[nl])
        #     enc_outputs, enc_states = tf.nn.dynamic_rnn(enc_cell, enc_inputs[nl], initial_state = state_init, dtype=tf.float32, time_major=True)
        #
        #  enc_final_state.append(enc_outputs) "not sure if append is the correct function"  # without LSTMStateTuple(enc_outputs[nl][-1]), since here the cell is GRU

# FW_G:dec

    # def output_deconv(inputs): # the projection layer of output, should be from 3 pooling layer to input layer
    #
    #     outputs = deconv2d(dec_outputs, )
    #
    #     return
    #
    #
    # for step in range(len(feature_maps)):
    #
    #     if step == 0:
    #
    #         for nl in range(nlayers):
    #
    #             dec_cell = ConvGRUCell(shape[nl], filters[nl], kernel[nl])
    #
    #             if nl > 0:
    #
    #                 dec_inputs = tf.concat(tf.zeros_like(feature_maps[nl]), dec_outputs[],1)
    #
    #                 dec_outputs, dec_states = tf.nn.dynamic_rnn(dec_cell, dec_inputs, initial_state=enc_final_state[nl], dtype = tf.float32, time_major=True, sequence_length=1)
    #
    #             else:
    #
    #                 dec_inputs = tf.zeros_like(feature_maps[])
    #
    #                 dec_outputs, dec_states = tf.nn.dynamic_rnn(dec_cell, dec_inputs, initial_state=enc_final_state[nl], dtype = tf.float32, time_major=True, sequence_length=1)
    #
    #                 final_outputs = output_deconv(dec_outputs)
    #
    #
    #     else:
    #
    #          for nl in range(nlayers):
    #
    #              dec_cell = ConvGRUCell(shape[nl], filters[nl], kernel[nl])
    #
    #              if nl > 0:
    #
    #                 dec_inputs = final_outputs[] # correspondingly output
    #
    #                 dec_outputs, dec_states = tf.nn.dynamic_rnn(dec_cell, dec_inputs, initial_state=enc_final_state[nl], dtype = tf.float32, time_major=True, sequence_length=1)
    #
    #              else:
    #
    #                 dec_inputs = final_outputs[-1]
    #
    #                 dec_outputs, dec_states = tf.nn.dynamic_rnn(dec_cell, dec_inputs, initial_state=enc_final_state[nl], dtype = tf.float32, time_major=True, sequence_length=1)
    #
    # return dec_outputs, final_dec_outputs




#model.saver = tf.train.Saver()

# train with wgan-gp


# decoder_lengths = len(feature_maps)
# # FW_G:dec
# for nl in range(nlayers):
#
#     dec_cell = ConvGRUCell(shape[nl], filters[nl], kernel[nl])
#
#     def loop_dec_init():
#
#         initial_elements_finished = (0 >= decoder_lengths)  # all False at the initial step
#
#         if nl>0:
#             initial_input = tf.zeros_like()
#         else:
#             initial_input = tf.zeros_like()
#             initial_cell_state = enc_final_state[nl]
#             initial_cell_output = None
#             initial_loop_state = None
#
#         return (initial_elements_finished, initial_input, initial_cell_state, initial_cell_output,initial_loop_state)
#
#     def loop_dec_airing(time, previous_output, previous_state, previous_loop_state):
#
#         def get_next_input(): # the deconv feature maps in previous step,
#                               # the output projection, do not forget residual...
#             return next_input
#
#         elements_finished = (time >= decoder_lengths) # descirbing if the seq are ended
#         finished = tf.reduce_all(elements_finished)
#         input = tf.cond(finished, lambda: , get_next_input)
#         state = previous_state
#         output = previous_output
#         loop_state = None
#
#         return(elements_finished, input, state, output, loop_state)
#
#     def loop_dec_fn(time, previous_output, previous_state, previous_loop_state):
#
#         if previous_state is None: # time == 0
#             assert previous_output is None and previous_state is None
#                 return loop_dec_init()
#         else:
#             return loop_dec_airing(time, previous_output, previous_state, previous_loop_state)
#
#     dec_outputs, dec_states, _ = tf.nn.raw_rnn(dec_cell, loop_fn)
#
#     if nl == nlayers:
#         # deconv with residual from dec_outputs[nl]; save the output for bw G

# BW G:
# for nl in range(nlayers):
#
#     enc_cell = ConvGRUCell(shape[nl], filters[nl], kernel[nl])
#
#     if nl > 0:
#         enc_inputs = tf.concat(feature_maps[nl], enc_outputs[nl-1], 1)
#         #enc_cell = ConvGRUCell(shape[nl], filters[nl], kernel[nl])
#         enc_outputs, enc_states = tf.nn.dynamic_rnn(enc_cell, enc_inputs[nl], dtype=tf.float32, time_major=True)
#     else:
#         enc_inputs = feature_maps[nl]
#         #enc_cell = ConvGRUCell(shape[nl], filters[nl], kernel[nl])
#         enc_outputs, enc_states = tf.nn.dynamic_rnn(enc_cell, enc_inputs[nl], dtype=tf.float32, time_major=True)
#
#     enc_final_state.append(enc_outputs) "not sure if append is the correct function" #without LSTMStateTuple(enc_outputs[nl][-1]), since here the cell is GRU
#
# for step in range(len(feature_maps)):
#
#     if step == 0:
#
#         for nl in range(nlayers):
#
#             dec_cell = ConvGRUCell(shape[nl], filters[nl], kernel[nl])
#
#             if nl > 0:
#
#                 dec_inputs = tf.concat(tf.zeros_like(feature_maps[nl]), dec_outputs[],1)
#
#                 dec_outputs, dec_states = tf.nn.dynamic_rnn(dec_cell, dec_inputs, initial_state=enc_final_state[nl], dtype = tf.float32, time_major=True, sequence_length=1)
#
#             else:
#
#                 dec_inputs = tf.zeros_like(feature_maps[])
#
#                 dec_outputs, dec_states = tf.nn.dynamic_rnn(dec_cell, dec_inputs, initial_state=enc_final_state[nl], dtype = tf.float32, time_major=True, sequence_length=1)
#
#                 final_outputs = output_deconv(dec_outputs)
#
#
#     else:
#
#         for nl in range(nlayers):
#
#             dec_cell = ConvGRUCell(shape[nl], filters[nl], kernel[nl])
#
#             if nl > 0:
#
#                 dec_inputs = final_outputs[] # correspondingly output
#
#                 dec_outputs, dec_states = tf.nn.dynamic_rnn(dec_cell, dec_inputs, initial_state=enc_final_state[nl], dtype = tf.float32, time_major=True, sequence_length=1)
#
#             else:
#
#                 dec_inputs = final_outputs[-1]
#
#                 dec_outputs, dec_states = tf.nn.dynamic_rnn(dec_cell, dec_inputs, initial_state=enc_final_state[nl], dtype = tf.float32, time_major=True, sequence_length=1)














