import os, sys, json
import numpy as np
import tensorflow as tf
import argparse
sys.path.append('.') 
sys.path.append('../datahub/') 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from chi_lib.FileFilter import FileFilter
from chi_lib.ProgressBar import ProgressBar
from chi_lib.library import *
from utils.utils import *

from blocks.data.JsonReader import JsonReader
from blocks.features.FormBowFeature import FormBowFeature
from blocks.features.TextlineCoordinateFeature import TextlineCoordinateFeature
from blocks.features.HeuristicGraphAdjMat import HeuristicGraphAdjMat
from blocks.labels.TextlineFormalKeyLabel import TextlineFormalKeyLabel
from blocks.neural_net.EthanGraphEmbedding import EthanGraphEmbedding
from blocks.neural_net.NonlocalEmbedding import NonlocalEmbedding
from blocks.neural_net.FeatureEmbedding import FeatureEmbedding
#from blocks.util.Cache import Cache
from blocks.loss.CrossEntropyLoss import CrossEntropyLoss
from blocks.optimize.AdamOptimizer import AdamOptimizer
from blocks.validate.ClassificationValidator import ClassificationValidator

from blocks.util.CheckpointBackup import CheckpointBackup
from blocks.util.SamplesFilter import SamplesFilter
from blocks.util.ExecutedLogExporter import ExecutedLogExporter
from blocks.util.ExecutedPickleExporter import ExecutedPickleExporter
from blocks.util.BlocksRunner import BlocksRunner



if __name__ == '__main__':
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--data', help='Data samples directory path', required=True)
	parser.add_argument('--corpus', help='Corpus file path', required=True)
	parser.add_argument('--classes', help='Classes file path', required=True)
	parser.add_argument('--train', help='Training file names .lst path', required=True)	
	parser.add_argument('--val', help='Validating file names .lst path', required=True)	
	parser.add_argument('--res', help='Result directory path', required=True)	
	parser.add_argument('--log_delay', help='Backup delay', default=5)	
	parser.add_argument('--epochs', help='Number of epochs', default=300)	
	parser.add_argument('--batch', help='Batch size', default=4)	
	parser.add_argument('--lr', help='Learning rate', default=0.001)	
	args = parser.parse_args()

	data_dir = args.data
	corpus = load_corpus(args.corpus)	
	classes = load_classes(args.classes)	
	log_delay = int(args.log_delay)
	epochs = int(args.epochs)
	batch_size = int(args.batch)
	learning_rate = float(args.lr)
	res_dir = args.res

	f_paths = loadValidFiles(data_dir, 'json', keepFilePath=True)
	file_filter = FileFilter(f_paths)

	sample_masks = {
		'train' : file_filter.mask_lst(args.train), 
		'val'   : file_filter.mask_lst(args.val)
	}

	# Data loading
	fileloader_block = JsonReader(f_paths)
	
	# Labels building
	value_labels = TextlineFormalKeyLabel(fileloader_block, classes=classes, class_types=['key', 'value'])

	# value_labels.build()
	# exit()

	# Features building
	bow = FormBowFeature(fileloader_block, corpus)
	coord = TextlineCoordinateFeature(fileloader_block)
	adjmats = HeuristicGraphAdjMat(fileloader_block)

	# Caching
	# bow = Cache(bow, cache_path=cache_dir)
	# coord = Cache(coord, cache_path=cache_dir)
	# adjmats = Cache(adjmats, cache_path=cache_dir)
	# value_labels = Cache(value_labels, cache_path=cache_dir)	

	# Embedding
	logit = EthanGraphEmbedding(bow, coord, adjmats)
	#logit = NonlocalEmbedding(bow, coord)
	embedding = FeatureEmbedding(logit, len(classes) * 2 - 1)
	
	# Loss building, filter only the training data
	train_embedding = SamplesFilter(embedding, sample_mask=sample_masks['train'])
	train_value_labels = SamplesFilter(value_labels, sample_mask=sample_masks['train'])
	loss = CrossEntropyLoss(train_embedding, train_value_labels, normalize=False)

	# loss.build()
	# exit()
	
	# Optimizer building 
	optimizer = AdamOptimizer(loss, batch_size=batch_size, learning_rate=learning_rate)

	# Validating
	# Filter only the validating data
	val_embedding = SamplesFilter(embedding, sample_mask=sample_masks['val'])
	val_value_labels = SamplesFilter(value_labels, sample_mask=sample_masks['val'])
	validator = ClassificationValidator(val_embedding, val_value_labels)

	# Run
	BlocksRunner(
		runnable_blocks=[
			# Optimize the model
			optimizer, 
			# Validate the result
			validator,
			# Backup checkpoint
			CheckpointBackup(res_dir),
			# Exporting log
			ExecutedLogExporter([optimizer, validator], res_dir),
			# Exporting executed output as pickle
			ExecutedPickleExporter([optimizer, validator], res_dir)],
		runnable_delays=[1, log_delay, log_delay, log_delay, log_delay],
		iterations=epochs
	).run()

