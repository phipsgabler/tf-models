#! /usr/bin/env python3

import tensorflow as tf
import numpy as np
import sys

from ffn_classifier import FFNClassifier

def ffn_example():
    with FFNClassifier(2, [10], 2, tf.train.AdamOptimizer(0.01)).with_session() as model:
        
        model.train_on_batch([[1.0, 2.0], [2.0, 1.0]], [0, 1])
        print(model.accuracy_on_batch([[1.0, 2.0], [2.0, 1.0]], [0, 1]))
        ckpt = model.save_checkpoint('bla.ckpt')
        print(ckpt)
        model.restore_from_checkpoint(ckpt)

if __name__ == '__main__':
    if len(sys.argv) == 1:
        sys.exit(0)
    elif sys.argv[1] == 'ffn':
        ffn_example()
