from __future__ import print_function

import os
import string
import zipfile
import numpy as np
import tensorflow as tf

dataset_path = 'dataset/'
dataset = 'text8.zip'

def read_data(dataset_path, dataset):
	'''Reading dataset as a string'''
	with zipfile.ZipFile(dataset_path + dataset) as f:
		text = tf.compat.as_str(f.read(f.namelist()[0]))

	return text

def split_dataset(text, validation_size):
	validation_text = text[:validation_size]
	train_text = text[validation_size:]

	return train_text, validation_text

def char2id(char):
	first_letter = ord(string.ascii_lowercase[0])

	if char in string.ascii_lowercase:
		return ord(char) - first_letter + 1
	elif char == ' ':
		return 0
	else:
		print('Unexpected character: %s' % char)
		return 0

def id2char(dictid):
	first_letter = ord(string.ascii_lowercase[0])

	if dictid > 0:
		return chr(dictid + first_letter - 1)
	else:
		return ' '

class BatchGenerator(object):
	def __init__(self, text, batch_size, num_unrollings):
		self._text = text
		self._text_size = len(text)
		self._batch_size = batch_size
		self._num_unrollings = num_unrollings
		segment = self._text_size // batch_size
		self._vocabulary_size = len(string.ascii_lowercase) + 1
		self._cursor = [offset * segment for offset in xrange(batch_size)]
		self._last_batch = self._next_batch()

	def _next_batch(self):
		'''Generate a single batch from the current cursor position in the data'''
		batch = np.zeros(shape=(self._batch_size, self._vocabulary_size), dtype=np.float)

		for b in xrange(self._batch_size):
			batch[b, char2id(self._text[self._cursor[b]])] = 1.0
			self._cursor[b] = (self._cursor[b] + 1) % self._text_size

		return batch

	def next(self):
		'''Generate the next array of batches from the data'''
		batches = [self._last_batch]

		for step in xrange(self._num_unrollings):
			batches.append(self._next_batch())

		self._last_batch = batches[-1]

		return batches

def characters(probabilities):
	'''Turn a 1-hot encoding or a probability distribution over the possible characters back'''
	return [id2char(c) for c in np.argmax(probabilities, 1)]

def batch2string(batches):
	'''Convert a sequence of batches back into their string representations'''
	s = [''] * batches[0].shape[0]

	for b in batches:
		s = [''.join(x) for x in zip(s, characters(b))]

	return s

if __name__ == '__main__':
	text = read_data(dataset_path, dataset)
	train_text, validation_text = split_dataset(text, validation_size=1000)

	train_batches = BatchGenerator(train_text, batch_size=64, num_unrollings=10)

	print(batch2string(train_batches.next()))
