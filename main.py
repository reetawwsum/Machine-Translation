import tensorflow as tf

from model import *
from config import *

def main(_):
	if FLAGS.train:
		model = Model()
		model.train()

if __name__ == '__main__':
	tf.app.run()
