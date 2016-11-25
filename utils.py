from __future__ import print_function

import os
import re
import gzip
import tarfile

import tensorflow as tf
from tensorflow.python.platform import gfile

class Dataset():
	'''Load dataset'''
	def __init__(self, config):
		self.config = config
		self.dataset_dir = config.dataset_dir
		self.en_vocabulary_size = config.en_vocabulary_size
		self.fr_vocabulary_size = config.fr_vocabulary_size
		self.file_name = os.path.join(config.dataset_dir, config.dataset)

		_PAD = b'_PAD'
		_GO = b'_GO'
		_EOS = b'_EOS'
		_UNK = b'_UNK'

		self._START_VOCAB = [_PAD, _GO, _EOS, _UNK]
		self._WORD_SPLIT = re.compile(b'([.,!?"\':;)(])')
		self._DIGIT_RE = re.compile(br'\d')

		self.load_dataset()

	def load_dataset(self):
		self.load()
		self.build_vocabularies()

	def gunzip_file(self, gz_path, new_path):
		print('Unpacking %s to %s' % (gz_path, new_path))
		with gzip.open(gz_path, 'rb') as gz_file:
			with open(new_path, 'wb') as new_file:
				for line in gz_file:
					new_file.write(line)

	def basic_tokenizer(self, sentence):
		words = []

		for space_separated_sentence in sentence.strip().split():
			words.extend(self._WORD_SPLIT.split(space_separated_sentence))

		return [w for w in words if w]

	def load(self):
		train_path = os.path.join(self.dataset_dir, 'giga-fren.release2.fixed')

		if not (gfile.Exists(train_path + '.fr') and gfile.Exists(train_path + '.en')):
			try:
				with tarfile.open(self.file_name, 'r') as corpus_tar:
					print('Extracting tar file %s' % self.file_name)
					corpus_tar.extractall(self.dataset_dir)
			except IOError:
				print('Place the dataset into data folder')
				exit()

			self.gunzip_file(train_path + '.en.gz', train_path + '.en')
			self.gunzip_file(train_path + '.fr.gz', train_path + '.fr')

		self.train_path = train_path

	def build_vocabularies(self):
		en_vocabulary_path = os.path.join(self.dataset_dir, 'vocab%d.en' % self.en_vocabulary_size)
		en_data_path = self.train_path + '.en'

		if not gfile.Exists(en_vocabulary_path):
			self.build_vocabulary(en_vocabulary_path, en_data_path, self.en_vocabulary_size)

		fr_vocabulary_path = os.path.join(self.dataset_dir, 'vocab%d.fr' % self.fr_vocabulary_size)
		fr_data_path = self.train_path + '.fr'

		if not gfile.Exists(fr_vocabulary_path):
			self.build_vocabulary(fr_vocabulary_path, fr_data_path, self.fr_vocabulary_size)


	def build_vocabulary(self, vocabulary_path, data_path, max_vocabulary_size):
		print('Creating vocabulary %s from data %s' % (vocabulary_path, data_path))
		vocab = {}
		with gfile.GFile(data_path, 'rb') as f:
			counter = 0
			for line in f:
				counter += 1
				if not counter % 100000:
					print(' Processing line %d' % counter)

				line = tf.compat.as_bytes(line)
				tokens = self.basic_tokenizer(line)

				for w in tokens:
					word = self._DIGIT_RE.sub(b'0', w)
					if word in vocab:
						vocab[word] += 1
					else:
						vocab[word] = 1

			vocab_list = self._START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)

			if len(vocab_list) > max_vocabulary_size:
				vocab_list = vocab_list[:max_vocabulary_size]

			with gfile.GFile(vocabulary_path, 'wb') as vocab_file:
				for w in vocab_list:
					vocab_file.write(w + b'\n')
