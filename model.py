from __future__ import print_function

import tensorflow as tf

from ops import *
from utils import *

class Model:
	'''RNN LSTM neural network'''
	def __init__(self, config):
		self.config = config
		self.num_units = config.num_units
		self.num_hidden_layers = config.num_hidden_layers
		self.learning_rate = config.learning_rate
		self.input_keep_prob_value = config.input_keep_prob
		self.output_keep_prob_value = config.output_keep_prob
		self.epochs = config.epochs
		self.batch_size = config.batch_size
		self.num_unrollings = config.num_unrollings
		self.checkpoint_step = config.checkpoint_step
		self.input_size = len(string.ascii_lowercase) + 1
		self.output_size = self.input_size

		self.build_model()

	def inference(self):
		# Creating LSTM layer
		cell = tf.nn.rnn_cell.LSTMCell(self.num_units)
		cell = tf.nn.rnn_cell.DropoutWrapper(cell, self.input_keep_prob, self.output_keep_prob)
		cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.num_hidden_layers)

		# Creating Unrolled LSTM
		hidden, _ = tf.nn.dynamic_rnn(cell, self.data, dtype=tf.float32)

		# Creating output at each time step
		reshaped_hidden = tf.reshape(hidden, (-1, self.num_units))
		prediction = tf.nn.softmax(tf.matmul(reshaped_hidden, self.weight) + self.bias)

		self.prediction = prediction

	def loss_op(self):
		reshaped_prediction = tf.reshape(self.prediction, (self.batch_size, self.num_unrollings, self.output_size))
		cross_entropy = -tf.reduce_sum(self.target * tf.log(reshaped_prediction))

		self.loss = cross_entropy

	def train_op(self):
		optimizer = tf.train.AdamOptimizer(self.learning_rate)

		self.optimizer = optimizer.minimize(self.loss)

	def create_saver(self):
		saver = tf.train.Saver()

		self.saver = saver

	def build_model(self):
		self.graph = tf.Graph()

		with self.graph.as_default():
			# Creating placeholder for data and target
			self.data, self.target = placeholder_input(self.input_size, self.output_size)

			# Creating placeholder for LSTM dropout
			self.input_keep_prob = placeholder_dropout()
			self.output_keep_prob = placeholder_dropout()

			# Creating variables for output layer
			self.weight = weight_variable([self.num_units, self.output_size])
			self.bias = bias_variable([self.output_size])

			# Builds the graph that computes inference
			self.inference()

			# Adding loss op to the graph
			self.loss_op()

			# Adding train op to the graph
			self.train_op()

			# Creating saver
			self.create_saver()

	def train(self):
		with tf.Session(graph=self.graph) as self.sess:
			init = tf.initialize_all_variables()
			self.sess.run(init)
			print('Graph Initialized')

			train_batches = BatchGenerator(self.config)

			steps_in_one_epoch = ((train_batches.train_size / self.batch_size) / self.num_unrollings)

			for step in xrange(self.epochs * steps_in_one_epoch + 1):
				train_data = train_batches.next()

				feed_dict = {self.data: train_data[:, :self.num_unrollings], self.target: train_data[:, 1:], self.input_keep_prob: self.input_keep_prob_value, self.output_keep_prob: self.output_keep_prob_value}	

				_, l = self.sess.run([self.optimizer, self.loss], feed_dict=feed_dict)

				assert self.checkpoint_step % steps_in_one_epoch == 0, 'Checkpoint step should be an Epoch'

				if not step % self.checkpoint_step:
					epoch = step / self.checkpoint_step

					print('Loss at Epoch %d: %f' % (epoch, l))
