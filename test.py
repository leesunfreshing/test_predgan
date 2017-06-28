import tensorflow as tf
from cell import ConvGRUCell
#from ConvGRU_new import ConvGRUCell as C
from layers import deconv2d

batch_size = 32
timesteps = 100
shape = [448, 224]
kernel = [3, 3]
kernel_1 = [5, 5]
kernel_2 = [7, 7]
channels = 12
filter = 12

# Create a placeholder for videos.
inputs = tf.placeholder(tf.float32, [batch_size, timesteps] + shape + [channels])
#print(inputs)

# Make placeholder batch major again after RNN. (see https://github.com/tensorflow/tensorflow/pull/5142)
inputs = tf.transpose(inputs, (1, 0, 2, 3, 4))
#print(inputs)
# There's also a ConvGRUCell that is more memory efficient.

#from cell import ConvGRUCell
#with tf.variable_scope('cell_0'):

# def add_cell(shapes, filters, kernels, initializer, reuse=None):
#     return ConvGRUCell(shapes, filters, kernels, initializer=initializer, reuse=reuse)

with tf.variable_scope('cell_0'):
     cell_0 = ConvGRUCell(shape, filter, kernel, initializer=tf.truncated_normal_initializer(stddev=0.01))
     #print(cell_0)
     outputs, state = tf.nn.dynamic_rnn(cell_0, inputs, initial_state=cell_0.zero_state(inputs.shape[1].value, dtype=tf.float32), dtype=inputs.dtype, time_major=True)
     outputs = tf.concat([outputs, outputs], 2)
#print(outputs)
#
#
with tf.variable_scope('cell_1'):
     cell_1 = ConvGRUCell(shape, 3, kernel_1, initializer=tf.truncated_normal_initializer(stddev=0.01))
     outputs_1, state_1 = tf.nn.dynamic_rnn(cell_1, outputs, initial_state=cell_1.zero_state(outputs.shape[1].value, dtype=tf.float32), dtype=inputs.dtype, time_major=True)
with tf.variable_scope('cell_1', reuse=True):
     cell_3 = ConvGRUCell(shape, 3, kernel_1)#, reuse=True)
     #print(cell_1)

     outputs_3, state_3 = tf.nn.dynamic_rnn(cell_3, outputs, initial_state=state_1, dtype=inputs.dtype, time_major=True)
print(outputs_1)
print(state_1)
print(outputs_3)
print(state_3)
# cell_0 = ConvGRUCell(shape, filter, kernel, initializer=tf.truncated_normal_initializer(stddev=0.01))
# print(cell_0)
# outputs, state = tf.nn.dynamic_rnn(cell_0, inputs, dtype=inputs.dtype, time_major=True)
# outputs = tf.concat([outputs, outputs], 2)
# print(outputs)

# cell_1 = ConvGRUCell(shape, 3, kernel_1, initializer=tf.truncated_normal_initializer(stddev=0.01))
# print(cell_1)
# outputs_1, state_1 = tf.nn.dynamic_rnn(cell_1, outputs, dtype=inputs.dtype, time_major=True)  #
# print(outputs_1)

with tf.variable_scope('cell_1', reuse=True):
     cell_2 = ConvGRUCell(shape, 3, kernel_1)#, reuse=True)
# print(cell_1_2)
     outputs_2, state_2 = tf.nn.dynamic_rnn(cell_2, outputs, initial_state=state_1, dtype=inputs.dtype, time_major=True)
print(outputs_2)
print(state_2)
#if outputs_2 != outputs_1:
#print(outputs_2)
#print(outputs_2[0])
output_deconv = deconv2d(outputs_2[0], [32, 448, 448, 3])
shape = [output_deconv.shape[0], output_deconv.shape[1]*2, output_deconv.shape[2]*2, output_deconv.shape[3]]


# with tf.Session() as sess:
#      sess.run(tf.initialize_all_variables())
#      sess.run(outputs_1.name)
#      sess.run(outputs_1)
#      sess.run(outputs_2.name)
#      sess.run(outputs_2)


#print(shape)
#





#

out = []
for i in range(outputs.shape[0].value):
    out_temp = tf.nn.max_pool(outputs[i], ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    #print(out_temp)
    out.append(out_temp)
print(out)
out = tf.stack(out)
print(out.shape[0].value)
print(out)
