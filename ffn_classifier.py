import numpy as np
import tensorflow as tf
from model import Model, graph_property
from builders import fully_connected_layer


class FFNClassifier(Model):
    def __init__(self, n_input, hidden_layers, n_classes, optimizer):
        self.n_input = n_input
        self.hidden_layers = hidden_layers
        self.n_classes = n_classes
        self.optimizer = optimizer

        super().__init__()


    @graph_property
    def inputs(self):
        return tf.placeholder(tf.float32, shape = (None, self.n_input))

    
    @graph_property
    def targets(self):
        return tf.placeholder(tf.int64, shape = (None))


    @graph_property
    def logits(self):
        # construct hidden layers
        layers = [self.inputs]
        layer_sizes = [self.n_input] + self.hidden_layers

        for i in range(1, len(layer_sizes)):
            layers.append(fully_connected_layer('full{}'.format(i),
                                                layers[-1],
                                                layer_sizes[i-1],
                                                layer_sizes[i]))

        # the parts of the output layer
        return fully_connected_layer('logits', layers[-1], layer_sizes[-1], self.n_classes,
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





    
        


    


