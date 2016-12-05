from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf

import ops
import config

class Model():
	'''Sequence to sequence translation model'''
	def __init__(self):
		self.batch_size = config.FLAGS.batch_size
		self.num_units = config.FLAGS.num_units
		self.num_hidden_layers = config.FLAGS.num_hidden_layers
		self.learning_rate = config.FLAGS.learning_rate
		self.learning_rate_decay_factor = config.FLAGS.learning_rate_decay_factor
		self.max_gradient_norm = config.FLAGS.max_gradient_norm
		self.num_samples = config.FLAGS.num_samples
		self.buckets = config.BUCKETS

		# English to French translation
		if config.FLAGS.target_vocab == 'fr':
			self.source_vocab_size = config.FLAGS.en_vocabulary_size
			self.target_vocab_size = config.FLAGS.fr_vocabulary_size
		else:
			self.source_vocab_size = config.FLAGS.fr_vocabulary_size
			self.target_vocab_size = config.FLAGS.en_vocabulary_size

		self.build_model()

	def inference_and_loss(self):
		# Creating LSTM layer
		single_cell = tf.nn.rnn_cell.LSTMCell(self.num_units)
		cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * self.num_hidden_layers)

		# Creating embedding seq2seq function with attention
		seq2seq_f = ops.embedding_seq2seq_with_attention(cell, self.source_vocab_size, self.target_vocab_size, self.num_units, self.output_projection)

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
			self.encoder_inputs, self.decoder_inputs = ops.encoder_decoder_input_placeholder(self.buckets[-1][0], self.buckets[-1][1] + 1)

			# Creating placeholder for targets
			self.targets = ops.target_placeholder(self.decoder_inputs)

			# Creating placeholder for target weights
			self.target_weights = ops.target_weight_placeholder(self.buckets[-1][1] + 1)

			# Creating output projection and softmax loss function in order to handle large vocabulary
			self.output_projection, self.softmax_loss_function = ops.handle_large_vocabulary(self.num_samples, self.num_units, self.target_vocab_size)

			# Creating learning rate variable, learning rate decay op, and global step for train op
			self.learning_rate_var, self.learning_rate_decay_op, self.global_step = ops.get_more_hyperparameters(self.learning_rate, self.learning_rate_decay_factor)

			# Builds the graph that computes inference and loss
			self.inference_and_loss()

			# Adding train op to the graph
			self.train_op()

			# Creating saver
			self.create_saver()

	def train(self):
		with tf.Session(graph=self.graph) as self.sess:
			init = tf.initialize_all_variables()
			self.sess.run(init)
			print('Graph Initialised')
