[![No Maintenance Intended](http://unmaintained.tech/badge.svg)](http://unmaintained.tech/)

# tf-models

This repository contains a few implementations of TensorFlow models I've used so far (at university), cleaned
up and using a common interface.

I've never been a fan of just putting the whole TF code for a model in a couple of functions,
or worse, just into `main`.  [This site](http://danijar.com/structuring-your-tensorflow-models/) proposes
a nice way to organize the graph parts of a model, which I reuse here.  Additionally, I like to have all 
calls to `sess.run` also be handled by the model, and only provide "semantic" methods to the outside, like
[Keras models](https://keras.io/models/model/) do.  All this I tried to put into the [`ModelBase`](./model.py)
abstract class.

## Notes

- All tensors used are 64 bit now. I might parametrize this.
- There is a default implementation for calculating the accuracy on a test batch.  If you use this for a 
  regression model, it's you own fault...
- [`examples.py`](./examples.py) currently contains no real examples (and does _not_ use any sensible hyperparameters),
  it is just there to be able to test whether the implementation works at all on the TF side.

## ToDo

- Add an example of a convolutional autoencoder (since getting convolution/deconvolution to work requires 
  a lot of thinking, and autoencoders are cool).
- Use actual examples, not just stupid constants.
- More options for configuring the classes.
- Think of a better way of managing sessions.
