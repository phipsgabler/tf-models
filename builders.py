import numpy as np
import tensorflow as tf

def fully_connected_layer(name, inputs, n_in, n_out, activation = tf.nn.relu):
    with tf.variable_scope(name):
        W = tf.get_variable('W', initializer = tf.truncated_normal([n_in, n_out],
                                                                   stddev = 1.0 / np.sqrt(n_in)))
        b = tf.get_variable('b', initializer = tf.zeros(n_out))
        return activation(tf.matmul(inputs, W) + b)
