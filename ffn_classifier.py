import numpy as np
import tensorflow as tf

class FFNClassifier:
    def __init__(self, n_input, hidden_layers, n_classes, optimizer):
        with tf.Graph().as_default():
            self.inputs = tf.placeholder(tf.float32, shape = (None, n_input))
            self.classes = tf.placeholder(tf.int64, shape = (None))
            self.targets = tf.one_hot(self.classes, depth = n_classes)

            # construct hidden layers
            layers = [self.inputs]
            layer_sizes = [n_input] + hidden_layers

            for i in range(1, len(layer_sizes)):
                layers.append(self._fully_connected_layer('full{}'.format(i),
                                                          layers[-1],
                                                          layer_sizes[i-1],
                                                          layer_sizes[i]))

            # the parts of the output layer
            self.logits = self._fully_connected_layer(layers[-1], layer_sizes[-1], n_classes,
                                                      activation = lambda x: x)
            self.outputs = tf.nn.softmax(self.logits, name = 'outputs')
            self.predictions = tf.argmax(self.outputs, 1,
                                         name = 'predictions') # convert one-hot back to classes

            # loss & training
            self.global_step = tf.Variable(0, name = 'global_step', trainable = False)
            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(self.logits, self.targets,
                                                                         name = 'cross_entropy')
            self.loss = tf.reduce_mean(self.cross_entropy, name = 'loss')
            self.train = optimizer.minimize(self.loss, global_step = self.global_step)

            # classification accuracy
            correct_predictions = tf.equal(self.predictions, self.classes)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

            # general stuff
            self.session = tf.Session()
            self.saver = tf.train.Saver(max_to_keep = 1)




    def __enter__(self):
        self.session.__enter__()
        self.session.run(tf.initialize_all_variables())
        return self

    
    def __exit__(self, exc_type, exc_value, traceback):
        return self.session.__exit__(exc_type, exc_value, traceback)

        
    def _fully_connected_layer(self, name, inputs, n_in, n_out, activation = tf.nn.relu):
        with tf.variable_scope(name):
            W = tf.get_variable('W', initializer = tf.truncated_normal([n_in, n_out],
                                                                       stddev = 1.0 / np.sqrt(n_in)))
            b = tf.get_variable('b', initializer = tf.zeros(n_out))
            return activation(tf.matmul(inputs, W) + b)

    
    def train_on_batch(self, batch, targets):
        # batch: n_batch * n_feature
        # targets: n_batch (one-hot encoding is performed internally)
        return self.session.run(self.train, {self.inputs: batch,
                                             self.classes: targets})


    def test_on_batch(self, batch, targets):
        # batch: n_batch * n_feature
        # targets: n_batch (one-hot encoding is performed internally)
        return self.session.run(self.loss, {self.inputs: batch,
                                            self.classes: targets})

    def accuracy_on_batch(self, batch, targets):
        # batch: n_batch * n_feature
        # targets: n_batch (one-hot encoding is performed internally)        
        return self.session.run(self.accuracy, {self.inputs: batch,
                                                self.classes: targets})
    
    def predict_on_batch(self, batch):
        # batch: n_batch * n_feature
        print( batch.shape)
        return self.session.run(self.predictions, {self.inputs: batch})


    def save_checkpoint(self, save_path):
        return self.saver.save(self.session, save_path, self.global_step)

    
    def restore_from_checkpoint(self, save_path):
        self.saver.restore(self.session, save_path)
