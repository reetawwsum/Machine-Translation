from __future__ import print_function
from __future__ import absolute_import

from config import *
from utils.dataset import Dataset

class BatchGenerator():
	'''Generate Batches'''
	def __init__(self):
		config = FLAGS
		self.batch_size = config.batch_size

		self.load_dataset()

	def load_dataset(self):
		dataset = Dataset()

	def next(self):
		pass
