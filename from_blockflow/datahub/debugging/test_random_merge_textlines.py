# encoding: utf-8


import os, sys, json
sys.path.append('.') 

import glob                                                           
import cv2 
import numpy as np
import argparse
import shutil
import time

from collections import defaultdict
from frozendict import frozendict
from copy import deepcopy

from chi_lib.library import *
from chi_lib.Multiprocessor import Multiprocessor
from chi_lib.ProgressBar import ProgressBar

from augmenting.random_merge_textlines import random_merge_textlines
from augmenting.rotate_sample import rotate_sample
from visualizing.utils import visualize_multi_samples


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--path', help='Dataset directory path', required=True)
	parser.add_argument('--classes', help='Classes file path', required=True)
	parser.add_argument('--corpus', help='Corpus file path', required=True)
	parser.add_argument('--selected', nargs='+', help='Selected file names .lst paths', default=[])
	args = parser.parse_args()

	data_dir_path = args.path
	f_paths = loadValidFiles(data_dir_path, 'json', keepFilePath=True)
	f_paths = filter_file_paths_from_path_lists(f_paths, args.selected)

	classes = load_json(args.classes)
	corpus = load_json(args.corpus)
	
	samples = load_jsons(f_paths)

	for sample in samples[0:100]:
		start = time.clock()
		merged_sample = random_merge_textlines(sample, classes, corpus, max_dist_pct_w=0.03, merge_rate=0.5)
		print('Yo1', len(sample), time.clock() - start)

		#image = visualize_multi_samples([sample, merged_sample], max_width=1700)	
		#image.show()
	#database_map = get_classes_database(samples)
	#print(database_map)
	









