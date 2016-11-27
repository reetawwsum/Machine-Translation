import tensorflow as tf

def encoder_decoder_input_placeholder(encoder_input_range, decoder_input_range):
	encoder_inputs = []
	decoder_inputs = []

	for i in xrange(encoder_input_range):
		encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name='encoder{0}'.format(i)))

	for i in xrange(decoder_input_range):
		decoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name='decoder{0}'.format(i)))

	return encoder_inputs, decoder_inputs

def target_placeholder(decoder_inputs):
	return [decoder_inputs[i + 1] for i in xrange(len(decoder_inputs) - 1)]

def target_weight_placeholder(decoder_input_range):
	target_weights = []

	for i in xrange(decoder_input_range):
		target_weights.append(tf.placeholder(tf.float32, shape=[None], name='weight{0}'.format(i)))

	return target_weights
