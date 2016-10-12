from __future__ import print_function

import string
import tensorflow as tf

from character_model_input import *

num_unrollings = 10
batch_size = 64
vocabulary_size = len(string.ascii_lowercase) + 1
num_units = 64
num_steps = 7001

def run(param):
	# Building my graph
	graph = tf.Graph()

	with graph.as_default():
		# Creating placeholder for input and output sequences
		input_sequence = tf.placeholder(tf.float32, shape=[num_unrollings, batch_size, vocabulary_size])
		output_sequence = tf.placeholder(tf.float32, shape=[num_unrollings, batch_size, vocabulary_size])

		# Creating variables for output layer
		weight = tf.Variable(tf.truncated_normal([num_units, vocabulary_size], stddev=0.1))
		bias = tf.Variable(tf.constant(0.1, shape=[vocabulary_size]))

		# Creating LSTM layer
		cell = tf.nn.rnn_cell.LSTMCell(num_units)

		# Unrolled LSTM
		output, _ = tf.nn.dynamic_rnn(cell, input_sequence, dtype=tf.float32, time_major=True)

		# Calculating output at each time step
		output = tf.reshape(output, [-1, num_units])
		logits = tf.matmul(output, weight) + bias
		predictions = tf.nn.softmax(logits)

		# Adding loss op
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, output_sequence))

		# Adding optimizer
		optimizer = tf.train.AdamOptimizer()
		minimize = optimizer.minimize(loss)

	if param == 'training':
		# Training Character model
		with tf.Session(graph=graph) as sess:
			# Initializing all variables
			init = tf.initialize_all_variables()
			sess.run(init)
			print('Graph Initialized')

if __name__ == '__main__':
	run('training')