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
# from blocks.frozen.FrozenGraphEmbedding import FrozenGraphEmbedding
from blocks.frozen.FrozenGraphWeightVector import FrozenGraphWeightVector
from blocks.infer.ClassificationInferrer import ClassificationInferrer
from blocks.infer.ConfidenceInferrer import ConfidenceInferrer
#from blocks.util.Cache import Cache
from blocks.util.SamplesFilter import SamplesFilter
from blocks.util.BlocksRunner import BlocksRunner
from blocks.util.ExecutedPickleExporter import ExecutedPickleExporter

from blocks.infer.OneToOneTextlineInferrer import OneToOneTextlineInferrer


def infering(f_paths, corpus, classes, model_pb_path, res_dir):
	# Data loading
	fileloader_block = JsonReader(f_paths)

	# Features building
	bow = FormBowFeature(fileloader_block, corpus)
	coord = TextlineCoordinateFeature(fileloader_block)
	adjmats = HeuristicGraphAdjMat(fileloader_block)
	labels = TextlineFormalKeyLabel(fileloader_block, classes=classes, class_types=['value'])

	# Model embedding
	embedding = FrozenGraphWeightVector(bow, coord, adjmats, model_pb_path)

	# Inferring
	logit_inferrer = ClassificationInferrer(embedding)

	############################## Work here ##############################
	# Confidence infer
	conf_inferrer = ConfidenceInferrer(embedding, bow, labels)

	############################## Work here ##############################

	# Content inference
	# textline_inferrer = OneToOneTextlineInferrer(conf_inferrer, fileloader_block, classes, corpus)

	BlocksRunner(
		runnable_blocks=[
			# Inferring
			# logit_inferrer,
			# Confidence inference,
			conf_inferrer,
			# Mapping the inferred conf to textline
			# textline_inferrer,
			# Exporting inferences
			# ExecutedPickleExporter([conf_inferrer], res_dir=res_dir)
		]
	).run()


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

	for f_path in f_paths:
		f_res_dir = os.path.join(res_dir, getBasename(f_path).replace('.json', ''))
		infering([f_path], corpus, classes, model_pb_path, f_res_dir)