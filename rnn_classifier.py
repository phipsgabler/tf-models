import numpy as np
import tensorflow as tf
from model import ModelBase, graph_property
from builders import fully_connected_layer

class RNNClassifier(ModelBase):
    def __init__(self,
                 seq_length,
                 input_size,
                 n_classes,
                 recurrent_layer_size,
                 hidden_layer_sizes,
                 optimizer,
                 use_lstm = False):
        self.seq_length = seq_length
        self.input_size = input_size
        self.n_classes = n_classes
        self.recurrent_layer_size = recurrent_layer_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.optimizer = optimizer
        self.use_lstm = use_lstm
        
        super().__init__()
        

    @graph_property
    def inputs(self):
        return tf.placeholder(tf.float32, [None, self.seq_length, self.input_size])

    
    @graph_property
    def targets(self):
        return tf.placeholder(tf.int64, [None])

    
    @graph_property
    def recurrent_output(self):
        if self.use_lstm:
            cell = tf.nn.rnn_cell.LSTMCell(self.recurrent_layer_size, state_is_tuple = True)
        else:
            cell = tf.nn.rnn_cell.BasicRNNCell(self.recurrent_layer_size)
        
        rec_output, rec_state = tf.nn.dynamic_rnn(cell, self.inputs, dtype = tf.float32)
        return rec_output[:, -1, :] # extract only last output of recurrent part

    
    @graph_property
    def logits(self):
        hidden_layers = [self.recurrent_output]
        layer_sizes = [self.recurrent_layer_size] + self.hidden_layer_sizes

        for i in range(1, len(layer_sizes)):
            layer = fully_connected_layer('full{}'.format(i),
                                          hidden_layers[-1],
                                          layer_sizes[i-1],
                                          layer_sizes[i])
            hidden_layers.append(layer)

        return fully_connected_layer('logits',
                                     hidden_layers[-1],
                                     self.hidden_layer_sizes[-1],
                                     self.n_classes,
                                     activation = lambda x: x)


    @graph_property
    def predictions(self):
        outputs = tf.nn.softmax(self.logits, name = 'outputs')
        return tf.argmax(outputs, 1, name = 'predictions') # convert one-hot back to classes

    
    @graph_property
    def loss(self):
        targets_one_hot = tf.one_hot(self.targets, depth = self.n_classes)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(self.logits,
                                                                targets_one_hot,
                                                                name = 'cross_entropy')
        return tf.reduce_mean(cross_entropy, name = 'loss')

    
    @graph_property
    def train(self):
        return self.optimizer.minimize(self.loss, global_step = self.global_step)
        

            
