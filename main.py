from __future__ import print_function

from utils import *
from model import *

flags = tf.app.flags
flags.DEFINE_integer('batch_size', 100, 'The size of training batch')
flags.DEFINE_integer('num_unrollings', 10, 'Input sequence length')
flags.DEFINE_string('dataset_dir', 'data', 'Directory name for the dataset')
flags.DEFINE_string('dataset', 'text8.zip', 'Name of dataset')
flags.DEFINE_integer('validation_size', 1000, 'Size of validation dataset')
flags.DEFINE_string('batch_dataset_type', 'train_dataset', 'Dataset used for generating training batches')
flags.DEFINE_string('validation_dataset_type', 'validation_dataset', 'Dataset used for validation')
FLAGS = flags.FLAGS

def main(_):
	# Validating flags
	assert FLAGS.num_unrollings > 0, 'Input greater sequence length should be greater than 0'

if __name__ == '__main__':
	tf.app.run()
