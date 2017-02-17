import numpy as np
import tensorflow as tf
from collections import namedtuple

ConvLayer = namedtuple('ConvLayer', ['output', 'weights', 'filter'])


# see: http://datascience.stackexchange.com/q/15307
class ConvolutionalAutoencoder:
    
    def __init__(self,
                 input_shape,
                 filter_shapes,
                 optimizer = tf.train.RMSPropOptimizer(0.01)):

        super().__init__()

        self.channels = input_shape[2]
        self.batch_shape = input_shape[:2]
        self.filters = filter_shapes

        self.batch_input = tf.placeholder(tf.float32, (None, *input_shape))
        self.batch_size = tf.shape(self.batch_input)[0]

        
        # construct encoding layers
        encoding_layers = [ConvLayer(self.batch_input, None, {'channels': self.channels})]
        
        with tf.variable_scope('encoding'):
            for i, f in enumerate(filter_shapes):
                previous_layer = encoding_layers[-1]
                
                with tf.variable_scope('conv{}'.format(i)):
                    conv, w = self._conv2d_layer(previous_layer.output,
                                                 previous_layer.filter['channels'],
                                                 f['size'], f['stride'], f['channels'])
                    encoding_layers.append(ConvLayer(conv, w, f))

        self.encode = encoding_layers[-1].output

        # construct decoding layers
        decoding_layers = [self.encode]
        with tf.variable_scope('decoding'):
            for i in range(len(encoding_layers) - 1, 0, -1):
                with tf.variable_scope('deconv{}'.format(i)):
                    previous_layer = decoding_layers[-1]
                    corresponding_layer = encoding_layers[i]
                    corresponding_previous_layer = encoding_layers[i - 1]
                    corresponding_filter = corresponding_layer.filter
                    corresponding_out_shape = (tf.shape(corresponding_previous_layer.output)[1],
                                               tf.shape(corresponding_previous_layer.output)[2])
                    corresponding_out_channels = corresponding_previous_layer.filter['channels']
                    
                    deconv = self._deconv2d_layer(previous_layer,
                                                  corresponding_layer.weights,
                                                  stride = corresponding_filter['stride'],
                                                  out_shape = corresponding_out_shape,
                                                  out_channels = corresponding_out_channels)
                    decoding_layers.append(deconv)
        
        self.decode = decoding_layers[-1]

        self.loss = tf.reduce_mean(tf.square(self.batch_input - self.decode))
        self.optimize = optimizer.minimize(self.loss)

        self.session = tf.Session()
        self.saver = tf.train.Saver()

        
    def _conv2d_layer(self, input, in_channels, size, stride, out_channels):
        w_stddev = 1.0 / np.sqrt(in_channels)
        w = tf.get_variable('w',
                            shape = (size, size, in_channels, out_channels),
                            initializer = tf.truncated_normal_initializer(stddev = w_stddev))
        b = tf.get_variable('b', shape = (1,), initializer = tf.constant_initializer(0))
        
        activation = tf.nn.conv2d(input, w,
                                  strides = (1, stride, stride, 1),
                                  padding = 'VALID')

        return tf.nn.relu(activation + b), w

    def _deconv2d_layer(self, input, shared_w, stride, out_shape, out_channels):
        b = tf.get_variable('b', shape = (1,), initializer = tf.constant_initializer(0))

        activation = tf.nn.conv2d_transpose(
            input,
            filter = shared_w,
            output_shape = (self.batch_size, *out_shape, out_channels),
            strides = (1, stride, stride, 1),
            padding = 'VALID')
        
        return tf.nn.relu(activation + b)
    
        
    def __enter__(self):
        self.session.__enter__()
        self.session.run(tf.initialize_all_variables())
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self.session.__exit__(exc_type, exc_value, traceback)

        
    def encode(self, batch):
        return self.session.run(self.encode, {self.batch_input: batch})

    def decode(self, batch):
        pass

    
    def train_on_batch(self, batch):
        loss, _, decoded = self.session.run([self.loss, self.optimize, self.decode],
                                           {self.batch_input: batch})
        return loss, decoded
