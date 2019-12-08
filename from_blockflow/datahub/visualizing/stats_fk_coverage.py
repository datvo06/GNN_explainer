# encoding: utf-8


import os, sys, json
sys.path.append('.') 

import glob                                                           
import cv2 
import numpy as np
import argparse
import shutil

from chi_lib.library import *
from collections import defaultdict
from frozendict import frozendict
from chi_lib.Multiprocessor import Multiprocessor
from chi_lib.ProgressBar import ProgressBar


def get_closest_back_classes_path(cur_dir, max_step=5):
	while max_step > 0:
		max_step -= 1
		class_path = os.path.join(cur_dir, 'classes.json')
		if os.path.exists(class_path):
			return class_path
		cur_dir = getParentPath(cur_dir)
	return None

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--path', help='Dataset directory path', required=True)
	parser.add_argument('--selected', nargs='+', help='Selected samples "to generate stats" lst paths', default=[])
	args = parser.parse_args()

	dataDirPath = args.path
	fPaths = loadValidFiles(dataDirPath, 'json', keepFilePath=True)
	fPaths = filter_file_paths_from_path_lists(fPaths, args.selected)
	sample_dicts = load_jsons(fPaths)

	temp = get_classes_database(sample_dicts)
	pair_tuples = [k.split('#') for k in temp]
	keys = set([kt[0] for kt in pair_tuples])
	types = set([kt[1] for kt in pair_tuples if len(kt) > 1])

	split_data = defaultdict(lambda : {})
	for fk_type, fk_data in temp.items():
		for fk in fk_data:
			split_data[fk_type][fk] = fk_data[fk]

	for fk_type in split_data:
		data = split_data[fk_type]
		logTracker.log('\nFK list for type "' + str(fk_type) + '"')
		logTracker.log('-' * 50)
		logTracker.log('Count\tFormal key')
		logTracker.log('-' * 50)
		item_count = 0
		for field in sorted(data):
			item_count += len(data[field])
			logTracker.log(str(len(data[field])) + '\t' + str(field))
		logTracker.log('-' * 50)
		logTracker.log('Total keys: ' + str(len(data)))
		logTracker.log('Total items: ' + str(item_count))
		logTracker.log('-' * 50)
		logTracker.log('')


	max_step = 5
	class_path = get_closest_back_classes_path(dataDirPath, max_step)
	classes = []
	if not class_path is None:
		logTracker.log('Found class path at ' + class_path)
		with open(class_path, 'r', encoding='utf8') as f:
			classes = json.loads(f.read())
	else:
		logTracker.log('Cannot find any classes.json with ' + str(max_step) + ' parent directories backward')

	classes = list(set(classes))
	if 'None' in classes:
		classes.remove('None')
		classes = ['None'] + classes

	logTracker.log('Classes     : ' + str(sorted(classes)))
	logTracker.log('Formal keys : ' + str(sorted(keys)))
	logTracker.log('Types       : ' + str(sorted(types)))
	logTracker.log('-' * 50)
	logTracker.log('Total FK-Type pairs : ' + str(len(temp)))
	logTracker.log('Total formal keys   : ' + str(len(keys)))
	logTracker.log('Total types         : ' + str(len(types)))
	logTracker.log('Total class         : ' + str(len(classes)))








