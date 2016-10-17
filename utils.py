from __future__ import print_function

import os
import string
import zipfile
import numpy as np
import tensorflow as tf

from ops import *

class Dataset:
	'''Load dataset'''
	def __init__(self, config, dataset_type):
		self.config = config
		self.dataset_type = dataset_type
		self.file_name = os.path.join(config.dataset_dir, config.dataset)
		self.validation_size = config.validation_size

		self.load_dataset()

	def load_dataset(self):
		self.load()
		train_text, validation_text = self.split()

		if self.dataset_type == 'train_dataset':
			self.data = train_text
		else:
			self.data = validation_text

	def load(self):
		'''Reading dataset as a string'''
		with zipfile.ZipFile(self.file_name) as f:
			text = tf.compat.as_str(f.read(f.namelist()[0]))

			self.text = text

	def split(self):
		validation_text = self.text[:self.validation_size]
		train_text = self.text[self.validation_size:]

		return train_text, validation_text

class BatchGenerator():
	'''Generate train batches'''
	def __init__(self, config):
		self.config = config
		self.batch_size = config.batch_size
		self.num_unrollings = config.num_unrollings
		self.input_size = len(string.ascii_lowercase) + 1
		self.batch_dataset_type = config.batch_dataset_type

		self.load_dataset()

		assert len(self.train_data) % self.batch_size == 0, 'Train size should be divisible by batch size'
		segment = len(self.train_data) / self.batch_size

		assert segment > self.num_unrollings, 'Segment (train size/batch size) should be greater than num_unrollings'
		assert segment % self.num_unrollings == 0, 'Segment (train size/batch size) should be divisble by num_unrollings'

		self.cursor = [offset * segment for offset in xrange(self.batch_size)]

	def load_dataset(self):
		dataset = Dataset(self.config, self.batch_dataset_type)
		self.train_data = dataset.data

	def sequence(self, position):
		'''Generate a sequence from a cursor position'''
		sequence = np.zeros(shape=(self.num_unrollings + 1, self.input_size), dtype=np.float)

		for i in xrange(self.num_unrollings + 1):
			sequence[i, char2id(self.train_data[self.cursor[position]])] = 1.0
			self.cursor[position] = (self.cursor[position] + 1) % len(self.train_data)

		return sequence

	def next(self):
		'''Generate next batch from the data'''
		batch = []

		for position in xrange(self.batch_size):
			batch.append(self.sequence(position))

		return np.array(batch)
		