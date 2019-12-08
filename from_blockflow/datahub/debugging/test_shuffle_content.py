# encoding: utf-8


import os, sys, json
sys.path.append('.') 

import glob                                                           
import cv2 
import numpy as np
import argparse
import shutil

from collections import defaultdict
from frozendict import frozendict
from copy import deepcopy

from chi_lib.library import *
from chi_lib.Multiprocessor import Multiprocessor
from chi_lib.ProgressBar import ProgressBar

from augmenting.shuffle_textlines_content import shuffle_textlines_content
from augmenting.rotate_sample import rotate_sample
from visualizing.utils import visualize_multi_samples


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--path', help='Dataset directory path', required=True)
	parser.add_argument('--classes', help='Classes file path', required=True)
	parser.add_argument('--selected', nargs='+', help='Selected file names .lst paths', default=[])
	args = parser.parse_args()

	data_dir_path = args.path
	f_paths = loadValidFiles(data_dir_path, 'json', keepFilePath=True)
	f_paths = filter_file_paths_from_path_lists(f_paths, args.selected)

	classes = load_json(args.classes)
	samples = load_jsons(f_paths)

	database_map = get_classes_database(samples)
	for sample in samples[0:5]:
		shuffle_sample = shuffle_textlines_content(sample, database_map, extend_percentage=0.2)
		#print(sample)
		print('Yo1', len(sample))
		#print(shuffle_sample)
		#image = visualize_multi_samples([sample, shuffle_sample], max_width=1700)	
		#image.show()
	#database_map = get_classes_database(samples)
	#print(database_map)
	









