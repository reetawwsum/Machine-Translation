from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf

from config import *
from model import Model

def main(_):
	if FLAGS.train:
		model = Model()
		model.train()

if __name__ == '__main__':
	tf.app.run()
