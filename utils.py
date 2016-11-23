from __future__ import print_function

import os
import tarfile
import gzip

from tensorflow.python.platform import gfile

class Dataset():
	'''Load dataset'''
	def __init__(self, config):
		self.config = config
		self.dataset_dir = config.dataset_dir
		self.file_name = os.path.join(config.dataset_dir, config.dataset)

		self.load_dataset()

	def load_dataset(self):
		self.load()
		self.build_vocabulary()

	def gunzip_file(self, gz_path, new_path):
		print('Unpacking %s to %s' % (gz_path, new_path))
		with gzip.open(gz_path, 'rb') as gz_file:
			with open(new_path, 'wb') as new_file:
				for line in gz_file:
					new_file.write(line)

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

			self.gunzip_file(train_path + '.fr.gz', train_path + '.fr')
			self.gunzip_file(train_path + '.en.gz', train_path + '.en')

		self.train_path = train_path

	def build_vocabulary(self):
		print('Building vocabulary')
