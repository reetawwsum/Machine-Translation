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

def handle_large_vocabulary(num_samples, size, target_vocab_size):
	output_projection = None
	softmax_loss_function = None

	if num_samples > 0 and num_samples < target_vocab_size:
		w = tf.get_variable('proj_w', [size, target_vocab_size])
		w_t = tf.transpose(w)
		b = tf.get_variable('proj_b', [target_vocab_size])

		output_projection = (w, b)

		def sampled_loss(inputs, labels):
			local_w_t = tf.cast(w_t, tf.float32)
			local_b = tf.cast(b, tf.float32)
			local_inputs = tf.cast(inputs, tf.float32)
			labels = tf.reshape(labels, [-1, 1])

			return tf.cast(tf.nn.sampled_softmax_loss(local_w_t, local_b, local_inputs, labels, num_samples, target_vocab_size), tf.float32)

		softmax_loss_function = sampled_loss

	return output_projection, softmax_loss_function
