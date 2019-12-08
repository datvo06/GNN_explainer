# encoding: utf-8


import os, sys, json
sys.path.append('.') 

import glob                                                           
import cv2 
import numpy as np
import argparse
import shutil
import random

from collections import defaultdict
from frozendict import frozendict
from copy import deepcopy

from chi_lib.library import *
from chi_lib.Multiprocessor import Multiprocessor
from chi_lib.ProgressBar import ProgressBar
from augmenting.rotate_sample import rotate_sample
from augmenting.shuffle_textlines_content import shuffle_textlines_content
from augmenting.random_merge_textlines import random_merge_textlines
from visualizing.utils import visualize_multi_samples


def rotate_records(records, min_degree, max_degree):
	res = []
	logTracker.log('Rotating ' + str(len(records)) + ' samples from +-' + str(min_degree) + ' to ' + str(max_degree))
	progress = ProgressBar('Rotating', len(records))
	for record in records:
		sample, f_name = record
		degrees = np.random.randint(min_degree, max_degree)
		degrees *= [-1, 1][np.random.randint(0, 1)]
		new_name = f_name.replace('.json', '#rotate' + str(degrees) + '.json')
		res.append((rotate_sample(sample, degrees), new_name))
		progress.increase()
	progress.done()
	return res


def random_merged_textlines_records(records, classes, corpus, max_dist_pct_w, merge_rate):
	res = []
	logTracker.log('Random merging textlines for ' + str(len(records)) + ' samples')
	progress = ProgressBar('Merging', len(records))
	for record in records:
		sample, f_name = record
		new_name = f_name.replace('.json', '#merge.json')
		merged_sample = random_merge_textlines(sample, classes=classes, corpus=corpus, max_dist_pct_w=max_dist_pct_w, merge_rate=merge_rate)
		res.append((merged_sample, new_name))
		progress.increase()
	progress.done()
	return res


def shuffle_records_content(records, classes, extend_percentage):
	samples = [record[0] for record in records]
	database_map = get_classes_database(samples)
	res = []
	logTracker.log('Shuffle textline content for ' + str(len(records)) + ' samples')
	progress = ProgressBar('Shuffling', len(records))
	for record in records:
		sample, f_name = record
		new_name = f_name.replace('.json', '#shuffle.json')
		shuffled_sample = shuffle_textlines_content(sample, database_map=database_map, extend_percentage=extend_percentage)
		res.append((shuffled_sample, new_name))
		progress.increase()
	progress.done()
	return res	


def save_records(records, res_path):
	createDirectory(res_path)
	logTracker.log('Saving ' + str(len(records)) + ' samples to ' + res_path)
	progress = ProgressBar('Saving', len(records))
	for record in records:
		sample, f_name = record
		with open(os.path.join(res_path, f_name), 'w+', encoding='utf8') as f:
			f.write(json.dumps(sample, ensure_ascii=False, indent=4))
		progress.increase()
	progress.done()


def convert_to_records(json_paths):
	samples = load_jsons(json_paths)
	return [(samples[i], getBasename(json_paths[i])) for i in range(len(json_paths))]


def augment_records(records, classes, corpus):
	# Add original QA samples
	augmented_records = list(records)

	# Shuffle the content based on standard QA input first
	augmented_records += shuffle_records_content(augmented_records, classes=classes, extend_percentage=0.2)
	
	# Slightly rotate the samples (including shuffled content)
	augmented_records += rotate_records(augmented_records, min_degree=2, max_degree=5)
	
	# Then, randomly merge the rotated samples, along with the original samples
	augmented_records += random_merged_textlines_records(augmented_records, classes, corpus, max_dist_pct_w=0.03, merge_rate=0.6)

	# Strongly rotate the randomly textline-merged samples
	augmented_records += rotate_records(augmented_records, min_degree=2, max_degree=20)
	return augmented_records


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--path', help='Dataset directory path', required=True)
	parser.add_argument('--classes', help='Classes file path', required=True)
	parser.add_argument('--corpus', help='Corpus file path', required=True)
	parser.add_argument('--selected', nargs='+', help='Selected file names .lst paths', default=[])
	parser.add_argument('--res', help='Exporting directory path', default=None)
	args = parser.parse_args()

	data_dir_path = args.path
	f_paths = loadValidFiles(data_dir_path, 'json', keepFilePath=True)
	f_paths = filter_file_paths_from_path_lists(f_paths, args.selected)

	classes = load_json(args.classes)
	corpus = load_json(args.corpus)
	
	res_path = args.res
	if res_path is None:
		cur_dir_path = getParentPath(data_dir_path)
		res_path = os.path.join(cur_dir_path, getBasename(data_dir_path) + '-augmented-' + getTodayDatetimeString()) 

	records = convert_to_records(f_paths)
	augmented_records = augment_records(records, classes, corpus)	

	logTracker.log('Total samples: ' + str(len(augmented_records)))
	save_records(augmented_records, res_path)
	#image = visualize_multi_samples([sample, rotated_sample], max_width=1700)	
	#image.show()
	#database_map = get_classes_database(samples)
	#print(database_map)
	









