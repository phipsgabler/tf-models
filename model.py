import tensorflow as tf
import abc
import functools

# the style of this stuff is adapted from http://danijar.com/structuring-your-tensorflow-models/
# and the interface of keras
# decorator trick: http://stackoverflow.com/a/42367262/1346276

class GraphProperty(property):
    pass

def graph_property(getter):
    property_name = getter.__name__
    attribute = '_cache_' + property_name

    @GraphProperty
    @functools.wraps(getter)
    def decorated(self):
        if not hasattr(self, attribute):
            with self._graph.as_default():
                with tf.variable_scope(property_name):
                    setattr(self, attribute, getter(self))
        return getattr(self, attribute)
    
    return decorated


class ModelBase(abc.ABC):
    def __init__(self):
        self._session = None
        self._graph = tf.Graph()
        
        # initialize all properties
        fields = (k for k, v in vars(type(self)).items() if isinstance(v, GraphProperty))
        for field in fields:
            getattr(self, field)

        with self._graph.as_default():
            self._saver = tf.train.Saver(max_to_keep = 1)


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
    

    @property
    @abc.abstractmethod
    def inputs(self):
        ...
        
    @property
    @abc.abstractmethod
    def targets(self):
        ...
    
    @property
    @abc.abstractmethod
    def predictions(self):
        ...
        
    @property
    @abc.abstractmethod
    def loss(self):
        ...

    @property
    @abc.abstractmethod
    def predictions(self):
        ...

    
    @graph_property
    def global_step(self):
        return tf.Variable(0, name = 'global_step', trainable = False)

    
    @graph_property
    def accuracy(self):
        correct_predictions = tf.equal(self.predictions, self.targets)
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float64))


    def train_on_batch(self, batch, targets):
        return self._session.run(self.train, {self.inputs: batch,
                                              self.targets: targets})
    
    def test_on_batch(self, batch, targets):
        return self._session.run(self.loss, {self.inputs: batch,
                                             self.targets: targets})
    
    def accuracy_on_batch(self, batch, targets):
        return self._session.run(self.accuracy, {self.inputs: batch,
                                                 self.targets: targets})
    
    def predict_on_batch(self, batch):
        return self._session.run(self.predictions, {self.inputs: batch})

    
    def save_checkpoint(self, checkpoint_path):
        return self._saver.save(self._session, checkpoint_path, self.global_step)
    
    def restore_from_checkpoint(self, checkpoint_path):
        self._saver.restore(self._session, checkpoint_path)
