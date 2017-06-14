import tensorflow as tf
from ConvGRU import ConvGRUCell
from layers import deconv2d
from layers import lrelu
from layers import linear
from layers import conv2d
import numpy as np
from ops import *
from utils import *

LAMBDA = 10 # Gradient penalty lambda hyperparameter
lamda = 200 # wights of the function 8 in wgan-gp

batch_size = 32
timesteps = 100
shape = [640, 480]
kernel = [3, 3]
channels = 3
filters = 12
nlayers = 4

def add_layer(input, state_init=None):

    #with tf.variable_scope('add_layer'):

        cell = ConvGRUCell(shape, filters, kernel, reuse=True)

        if state_init == None:
            state_init = cell.zero_state()
        else:
            state_init = state_init

        cell_output, cell_state = tf.nn.dynamic_rnn(cell, input, initial_state = state_init, dtype=tf.float32, time_major=True)

        return cell_output, cell_state

def add_output_layer(input):

    outputs = deconv2d(dec_outputs, )

    return

def gen(input, nlayers, reuse=False):

    with tf.variable_scope('inputs'):

        xs = tf.placeholder(tf.float32, [self._batch_size, self._time_steps, self._input_size], name='xs')
        ys = tf.placeholder(tf.float32, [self._batch_size, self._time_steps, self._output_size], name='ys')

    with tf.name_scope('gen'):

        with tf.variable_scope('gen_enc'):

            for nl in range(1, nlayers+1):

                if state_prev == None:

                    init = None

                else:

                    init = state_prev

               if nl == 1:

                  output, states = add_layer(xs, state_init=init )

               else:

                  output, states = add_layer(output[nl - 1])

        with tf.variable_scope('gen_dec'):

            for step in range(len(feature_maps)):

                if step == 0:

                    for nl in range(nlayers+1):

                        with tf.variable_scope('gen_layer'+fix):

                            oututput, states = add_layer(tf.zeros_like(), state_init=states[nl], sequence_length=1)

                else:

                    for nl in range(nlayers+1):

                        tf.get_variable_scope().reuse_variables()

                        output, states = add_layer(output[step][nl], state_init=prev_states[step][nl], sequence_length=1)

    return output, states


def disc(inputs,  reuse=False):

    with tf.name_scope('disc'):

        with tf.variable_scope(tf.get_variable_scope()) as scope:
             if reuse:
                scope.reuse_variables()
             else:
                assert scope.reuse == False

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

def pred_gen(seq,  reuse = False):
    return gen(seq)

def pred_disc(dec_outputs, reuse = False):
    return disc()

# loss
hidd_pred, gen_pred = pred_gen(feature_maps)
disc_pred = pred_disc(gen_pred)

def recon_gen(, state_init, reuse = False):
    return gen()

def recon_disc(, state_init, reuse = False):
    return disc()

_, gen_recon = recon_gen(dec_outputs, hidden_pred) # remember here is hidden_pred_reversed
disc_recon = recon_disc(gen_recon)

loss_pre = tf.reduce_mean(-pred_disc(gen_pred) ) + tf.reduce_mean(pred_disc(gt), reuse = True) + lamda * tf.reduce_mean(tf.abs(gen_recon - fw_inputs))# FW_G_output))
loss_disc = tf.reduce_mean(-recon_disc(gen_recon)) + tf.reduce_mean(recon_disc(inputs), reuse = True)

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














