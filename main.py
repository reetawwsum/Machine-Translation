from __future__ import print_function

import tensorflow as tf

from utils import *

flags = tf.app.flags
flags.DEFINE_integer('en_vocabulary_size', 40000, 'English vocabulary size')
flags.DEFINE_integer('fr_vocabulary_size', 40000, 'French vocabulary size')
flags.DEFINE_string('dataset', 'training-giga-fren.tar', 'Name of the dataset file')
flags.DEFINE_string('dataset_dir', 'data', 'Directory name for the dataset')
FLAGS = flags.FLAGS

def main(_):
	dataset = Dataset(FLAGS)

if __name__ == '__main__':
	tf.app.run()
