from __future__ import print_function
from __future__ import absolute_import

import config
from utils.dataset import Dataset

class BatchGenerator():
	'''Generate Batches'''
	def __init__(self):
		self.batch_size = config.FLAGS.batch_size

		self.load_dataset()

	def load_dataset(self):
		dataset = Dataset()

	def next(self):
		pass
