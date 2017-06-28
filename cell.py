import tensorflow as tf

class ConvGRUCell(tf.contrib.rnn.RNNCell):
  """A GRU cell with convolutions instead of multiplications. with strides = 1"""

  def __init__(self, shape, filters, kernel, initializer=None, activation=tf.tanh, normalize=True):#, reuse=None):
    self._filters = filters
    self._kernel = kernel
    self._initializer = initializer
    self._activation = activation
    self._size = tf.TensorShape([shape[0]/2, shape[1]] + [self._filters])
    self._normalize = normalize
    self._feature_axis = self._size.ndims
    #self._reuse = reuse

  @property
  def state_size(self):
    return self._size

  @property
  def output_size(self):
    return self._size

  def __call__(self, x_in, h):
    with tf.variable_scope(tf.get_variable_scope() or self.__class__.__name__):#, reuse=self._reuse):
      #print(tf.get_variable_scope().name)

      with tf.variable_scope('Gates'):#, reuse=self._reuse):
        x, x_extra = tf.split(x_in, 2, axis=1)
        channels = x.shape[-1].value
        channels_extra = x_extra.shape[-1].value
        inputs = tf.concat([x, x_extra, h], axis=self._feature_axis)
        n = channels + channels_extra + self._filters
        m = 3 * self._filters if self._filters > 1 else 3
        W = tf.get_variable('kernel', self._kernel + [n, m], initializer=self._initializer)
        y = tf.nn.convolution(inputs, W, 'SAME')#, strides=[2, 2], dilation_rate=(1, 1))
        if self._normalize:
          reset_gate, reset_extra_gate, update_gate = tf.split(y, 3, axis=self._feature_axis)
          reset_gate = tf.contrib.layers.layer_norm(reset_gate)
          reset_extra_gate = tf.contrib.layers.layer_norm(reset_extra_gate)
          update_gate = tf.contrib.layers.layer_norm(update_gate)
        else:
          y += tf.get_variable('bias', [m], initializer=tf.constant_initializer(1.0))
          reset_gate, reset_extra_gate, update_gate = tf.split(y, 3, axis=self._feature_axis)
        reset_gate, reset_extra_gate, update_gate = tf.sigmoid(reset_gate), tf.sigmoid(reset_extra_gate), tf.sigmoid(update_gate)

      with tf.variable_scope('Output'):#, reuse=self._reuse):
        inputs = tf.concat([x, x_extra, reset_gate * h], axis=self._feature_axis)
        n = channels + channels_extra + self._filters
        m = self._filters
        W = tf.get_variable('kernel', self._kernel + [n, m], initializer=self._initializer)
        y = tf.nn.convolution(inputs, W, 'SAME')#, strides=[2, 2], dilation_rate=(1, 1))
        if self._normalize:
          y = tf.contrib.layers.layer_norm(y)
        else:
          y += tf.get_variable('bias', [m], initializer=tf.constant_initializer(0.0))
        y = self._activation(y)
        output = update_gate * h + (1 - update_gate) * y

      return output, output
