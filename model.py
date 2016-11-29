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
		self.learning_rate_decay_factor = config.learning_rate_decay_factor
		self.max_gradient_norm = config.max_gradient_norm
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

	def inference_and_loss(self):
		# Creating LSTM layer
		single_cell = tf.nn.rnn_cell.LSTMCell(self.num_units)
		cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * self.num_hidden_layers)

		# Creating embedding seq2seq function with attention
		seq2seq_f = embedding_seq2seq_with_attention(cell, self.source_vocab_size, self.target_vocab_size, self.num_units, self.output_projection)

		# Creating outputs and losses using model with buckets
		self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(self.encoder_inputs, self.decoder_inputs, self.targets, self.target_weights, self.buckets, lambda x, y: seq2seq_f(x, y), self.softmax_loss_function)

	def train_op(self):
		params = tf.trainable_variables()
		self.gradient_norms = []
		self.updates = []
		
		opt = tf.train.GradientDescentOptimizer(self.learning_rate_var)

		for b in xrange(len(self.buckets)):
			gradients = tf.gradients(self.losses[b], params)
			clipped_gradients, norm = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
			self.gradient_norms.append(norm)
			self.updates.append(opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step))

	def create_saver(self):
		saver = tf.train.Saver()

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

			# Creating learning rate variable, learning rate decay op, and global step for train op
			self.learning_rate_var = tf.Variable(float(self.learning_rate), trainable=False, dtype=tf.float32)
			self.learning_rate_decay_op = self.learning_rate_var.assign(self.learning_rate_var * self.learning_rate_decay_factor)
			self.global_step = tf.Variable(0, trainable=False)

			# Builds the graph that computes inference and loss
			self.inference_and_loss()

			# Adding train op to the graph
			self.train_op()

			# Creating saver
			self.create_saver()

	def train(self):
		pass
