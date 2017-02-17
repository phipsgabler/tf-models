import numpy as np
import tensorflow as tf

class RNNClassifier:
    def __init__(self,
                 seq_length,
                 recurrent_neurons,
                 hidden_neurons,
                 optimizer,
                 use_lstm):

        self._session = None
        self._graph = tf.Graph()
        
        with self._graph.as_default():
            # inspired by: https://danijar.com/introduction-to-recurrent-networks-in-tensorflow/

            self.inputs = tf.placeholder(tf.int32, [None, seq_length])
            prepared_inputs = tf.expand_dims(tf.cast(self.inputs, tf.float32), -1)
            
            with tf.variable_scope('recurrent_layer'):
                if use_lstm:
                  cell = tf.nn.rnn_cell.LSTMCell(recurrent_neurons, state_is_tuple = True)
                else:
                  cell = tf.nn.rnn_cell.BasicRNNCell(recurrent_neurons)

                rec_output, rec_state = tf.nn.dynamic_rnn(cell, prepared_inputs, dtype = tf.float32)
                rec_last = rec_output[:, -1, :] # extract last output of recurrent part

            with tf.variable_scope('hidden_layers'):
                hidden_layers = [rec_last]
                hidden_layer_sizes = [recurrent_neurons] + hidden_neurons

                for i in range(1, len(hidden_layer_sizes)):
                    layer = self._fully_connected_layer('full{}'.format(i),
                                                        hidden_layers[-1],
                                                        hidden_layer_sizes[i-1],
                                                        hidden_layer_sizes[i])
                    hidden_layers.append(layer)
                    
                hidden_last = hidden_layers[-1]

            with tf.variable_scope('output_layer'):
                weight = tf.Variable(tf.truncated_normal([hidden_layer_sizes[-1], 1],
                                                         stddev = 1.0 / np.sqrt(hidden_layer_sizes[-1])))
                bias = tf.Variable(tf.constant(0.0))
                
                self.logits = tf.squeeze(tf.matmul(hidden_last, weight) + bias)
                self.outputs = tf.nn.sigmoid(self.logits)
                self.predictions = tf.round(self.outputs)

            with tf.variable_scope('training'):
                self.targets = tf.placeholder(tf.float32, [None])
                self.loss = tf.nn.sigmoid_cross_entropy_with_logits(self.logits, self.targets)

                self.global_step = tf.Variable(0, trainable = False)
                self.train = optimizer.minimize(self.loss, global_step = self.global_step)

                correct_predictions = tf.equal(self.predictions, self.targets)
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


    def train_on_batch(self, batch, targets):
        # batch: n_batch * seq_length
        # targets: n_batch
        return self._session.run(self.train, {self.inputs: batch,
                                              self.targets: targets})

    def test_on_batch(self, batch, targets):
        # batch: n_batch * seq_length
        # targets: n_batch
        return self._session.run(self.loss, {self.inputs: batch,
                                             self.targets: targets})

    def accuracy_on_batch(self, batch, targets):
        # batch: n_batch * seq_length
        # targets: n_batch
        return self._session.run(self.accuracy, {self.inputs: batch,
                                                 self.targets: targets})
    
    def predict_on_batch(self, batch):
        # batch: n_batch * seq_length
        return self._session.run(self.predictions, {self.inputs: batch})
    

    def with_session(model):
        session = tf.Session(graph = model._graph)
        
        class _sessionmanager:
            def __enter__(self):
                model._session = session
                model._session.__enter__()
                model._session.run(tf.initialize_all_variables())
                return model

            def __exit__(self, exc_type, exc_value, traceback):
                exit_value = model._session.__exit__(exc_type, exc_value, traceback)
                model._session = None
                return exit_value

        return _sessionmanager()


    def _fully_connected_layer(self, name, inputs, n_in, n_out, activation = tf.nn.relu):
        with tf.variable_scope(name):
            W = tf.get_variable('W', initializer = tf.truncated_normal([n_in, n_out],
                                                                       stddev = 1.0 / np.sqrt(n_in)))
            b = tf.get_variable('b', initializer = tf.zeros(n_out))
            return activation(tf.matmul(inputs, W) + b)
        

            
