from __future__ import print_function

import os
import tensorflow as tf

from ops import *
from utils import *

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
		self.num_samples = config.num_samples
		self.buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]

		# English to French translation
		if config.target_vocab == 'fr':
			self.source_vocab_size = config.en_vocabulary_size
			self.target_vocab_size = config.fr_vocabulary_size
		else:
			self.source_vocab_size = config.fr_vocabulary_size
			self.target_vocab_size = config.en_vocabulary_size

		self.build_model()

	def inference(self):
		pass

	def loss_op(self):
		pass

	def train_op(self):
		pass

	def create_saver(self):
		saver = tf.train.saver()

		self.saver = saver

	def build_model(self):
		self.graph = tf.Graph()

		with self.graph.as_default():
			# Creating placeholder for encoder and decoder inputs
			self.encoder_inputs, self.decoder_inputs = encoder_decoder_input_placeholder(self.buckets[-1][0], self.buckets[-1][1] + 1)

			# Creating placeholder for targets
			self.targets = target_placeholder(self.decoder_inputs)

			# Creating placeholder for target weights
			self.target_weights = target_weight_placeholder(self.buckets[-1][1] + 1)

			# Creating output projection and softmax loss function in order to handle large vocabulary
			self.output_projection, self.softmax_loss_function = handle_large_vocabulary(self.num_samples, self.num_units, self.target_vocab_size)
			

	def train(self):
		pass
