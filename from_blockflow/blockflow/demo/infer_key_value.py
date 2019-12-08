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
from blocks.data.JsonExporter import JsonExporter
from blocks.features.FormBowFeature import FormBowFeature
from blocks.features.TextlineCoordinateFeature import TextlineCoordinateFeature
from blocks.features.HeuristicGraphAdjMat import HeuristicGraphAdjMat
from blocks.labels.TextlineFormalKeyLabel import TextlineFormalKeyLabel
from blocks.frozen.FrozenGraphEmbedding import FrozenGraphEmbedding
from blocks.infer.ClassificationInferrer import ClassificationInferrer
from blocks.infer.OneToOneTextlineKeyValueInferrer import OneToOneTextlineKeyValueInferrer
from blocks.validate.ClassificationValidator import ClassificationValidator
#from blocks.util.Cache import Cache
from blocks.util.SamplesFilter import SamplesFilter
from blocks.util.ExecutedLogExporter import ExecutedLogExporter
from blocks.util.ExecutedPickleExporter import ExecutedPickleExporter
from blocks.util.BlocksRunner import BlocksRunner

	

if __name__ == '__main__':
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--data', help='Data samples directory path', required=True)
	parser.add_argument('--pb', help='Frozen model .pb file path', required=True)
	parser.add_argument('--corpus', help='Corpus file path', required=True)
	parser.add_argument('--classes', help='Classes file path', required=True)
	parser.add_argument('--selected', nargs='+', help='Selected file names .lst paths', default=[])	
	parser.add_argument('--res', help='Result directory path', default='inference')	
	args = parser.parse_args()

	data_dir = args.data
	model_pb_path = args.pb
	corpus = load_corpus(args.corpus)	
	classes = load_classes(args.classes)
	res_dir = os.path.join(args.res, getTodayDatetimeString() + '_inferences')
	createDirectory(res_dir)
	
	f_paths = loadValidFiles(data_dir, 'json', keepFilePath=True)
	f_paths = filter_file_paths_from_path_lists(f_paths, args.selected)

	# Data loading
	fileloader_block = JsonReader(f_paths)
	
	# Features building
	bow = FormBowFeature(fileloader_block, corpus)
	coord = TextlineCoordinateFeature(fileloader_block)
	adjmats = HeuristicGraphAdjMat(fileloader_block)
	labels = TextlineFormalKeyLabel(fileloader_block, classes=classes, class_types=['key', 'value'])

	# Model embedding
	embedding = FrozenGraphEmbedding(bow, coord, adjmats, model_pb_path)
	
	# Inferring
	logit_inferrer = ClassificationInferrer(embedding)
	
	# Content inference
	textline_inferrer = OneToOneTextlineKeyValueInferrer(logit_inferrer, fileloader_block, classes, corpus)

	# Statistics
	validator = ClassificationValidator(embedding, labels)
	
	BlocksRunner(
		runnable_blocks=[
			# Inferring
			logit_inferrer,
			# Mapping the inferred logit to textline
			textline_inferrer,
			# Exporting inferences
			JsonExporter([textline_inferrer], res_dir=res_dir), 
			# Validating 
			validator, 
			# Exporting log
			ExecutedLogExporter([validator], res_dir),
			# Exporting result (for confusion metrics)
			ExecutedPickleExporter([validator], res_dir)
		]
	).run()