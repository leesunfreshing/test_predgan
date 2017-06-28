import tensorflow as tf

class ConvGRUCell(tf.contrib.rnn.RNNCell):
  """A GRU cell with convolutions instead of multiplications. with strides = 1"""

  def __init__(self, shape, filters, kernel, initializer=None, activation=tf.tanh, normalize=True):#, reuse=None):
    self._filters = filters
    self._kernel = kernel
    self._initializer = initializer
    self._activation = activation
    self._size = tf.TensorShape(shape + [self._filters])
    self._normalize = normalize
    self._feature_axis = self._size.ndims
    #self._reuse = reuse

  @property
  def state_size(self):
    return self._size

  @property
  def output_size(self):
    return self._size

  def __call__(self, x, h):
    with tf.variable_scope(tf.get_variable_scope() or self.__class__.__name__):#, reuse=self._reuse):

      with tf.variable_scope('Gates'):#, reuse=self._reuse):
        channels = x.shape[-1].value
        inputs = tf.concat([x, h], axis=self._feature_axis)
        n = channels + self._filters
        m = 2 * self._filters if self._filters > 1 else 2
        W = tf.get_variable('kernel', self._kernel + [n, m], initializer=self._initializer)
        y = tf.nn.convolution(inputs, W, 'SAME')#, strides=[2, 2])
        if self._normalize:
          reset_gate, update_gate = tf.split(y, 2, axis=self._feature_axis)
          reset_gate = tf.contrib.layers.layer_norm(reset_gate)
          update_gate = tf.contrib.layers.layer_norm(update_gate)
        else:
          y += tf.get_variable('bias', [m], initializer=tf.ones_initializer())
          reset_gate, update_gate = tf.split(y, 2, axis=self._feature_axis)
        reset_gate, update_gate = tf.sigmoid(reset_gate), tf.sigmoid(update_gate)

      with tf.variable_scope('Output'):#, reuse=self._reuse):
        inputs = tf.concat([x, reset_gate * h], axis=self._feature_axis)
        n = channels + self._filters
        m = self._filters
        W = tf.get_variable('kernel', self._kernel + [n, m], initializer=self._initializer)
        y = tf.nn.convolution(inputs, W, 'SAME')#, strides=[2, 2])
        if self._normalize:
          y = tf.contrib.layers.layer_norm(y)
        else:
          y += tf.get_variable('bias', [m], initializer=tf.zeros_initializer())
        y = self._activation(y)
        output = update_gate * h + (1 - update_gate) * y
        output = tf.nn.max_pool(output, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

      return output, output
