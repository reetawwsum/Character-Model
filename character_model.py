from __future__ import print_function

import string
import tensorflow as tf

from character_model_input import *

dataset_path = 'dataset/'
dataset = 'text8.zip'

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
		optimizer = tf.train.AdamOptimizer().minimize(loss)

	# Getting dataset
	text = read_data(dataset_path, dataset)
	train_text, validation_text = split_dataset(text, validation_size=1000)

	if param == 'training':
		# Training Character model
		with tf.Session(graph=graph) as sess:
			# Initializing all variables
			init = tf.initialize_all_variables()
			sess.run(init)
			print('Graph Initialized')

			train_batches = BatchGenerator(train_text, batch_size, num_unrollings)
			average_loss = 0

			for step in xrange(num_steps):
				batches = train_batches.next()
				
				feed_dict = {input_sequence: batches[:num_unrollings], output_sequence: batches[1:]}

				_, l = sess.run([optimizer, loss], feed_dict=feed_dict)

				average_loss += l

				if step % 500 == 0:
					if step > 0:
						average_loss /= 500
						print('Average loss at step %d: %f' % (step, average_loss))

if __name__ == '__main__':
	run('training')