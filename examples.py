#! /usr/bin/env python3

import tensorflow as tf
import numpy as np
import sys

from ffn_classifier import FFNClassifier
from rnn_classifier import RNNClassifier


def ffn_example():
    with FFNClassifier(2, [10], 2, tf.train.AdamOptimizer(0.01)).with_session() as model:
        samples = [[1.0, 2.0], [2.0, 1.0]]
        targets = [0, 1]

        for _ in range(100):
            model.train_on_batch(samples, targets)
        print(model.accuracy_on_batch(samples, targets))
        
        # ckpt = model.save_checkpoint('bla.ckpt')
        # print(ckpt)
        # model.restore_from_checkpoint(ckpt)


def rnn_example():
    with RNNClassifier(10, 1, 2, 10, [10], tf.train.AdamOptimizer(0.01)).with_session() as model:
        samples = np.random.choice([1, 2], 20).reshape((2, 10, 1))
        targets = [0, 1]

        for _ in range(100):
            model.train_on_batch(samples, [0, 1])
        print(model.accuracy_on_batch(samples, [0, 1]))

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('Nothing happens...')
        sys.exit(0)
    elif sys.argv[1] == 'ffn':
        ffn_example()
    elif sys.argv[1] == 'rnn':
        rnn_example()
