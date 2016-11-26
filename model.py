from __future__ import print_function

import os
import tensorflow as tf

class Model():
	'''Sequence to sequence translation model'''
	def __init__(self, config):
		self.config = config
		self.batch_size = config.batch_size
		self.num_units = config.num_units
		self.num_hidden_layers = config.num_hidden_layers
		self.learning_rate = config.learning_rate
		self.en_vocabulary_size = config.en_vocabulary_size
		self.fr_vocabulary_size = config.fr_vocabulary_size
		self._buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]

		self.build_model()

	def inference(self):
		pass

	def loss_op(self):
		pass

	def train_op(self):
		pass

	def create_saver(self):
		pass

	def build_model(self):
		pass

	def train(self):
		pass
