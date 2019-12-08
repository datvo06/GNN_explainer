# encoding: utf-8


import os, sys, json
sys.path.append('.') 

import glob                                                           
import cv2 
import numpy as np
import argparse
import shutil
import re
import time

from copy import deepcopy
from chi_lib.library import *
from collections import defaultdict
from frozendict import frozendict
from chi_lib.Multiprocessor import Multiprocessor
from chi_lib.ProgressBar import ProgressBar
from normalizing.normalize_text import normalize_text


def get_corpus_map(samples, required_keys):
	key_type_corpus_map = defaultdict(lambda : defaultdict(lambda : set()))
	for sample in samples:
		for textline in sample:
			label_info = textline['label_info']
			fk = label_info['formal_key']
			fk_type = label_info['key_type']
			if fk in required_keys and fk_type in ['value', 'key']:
				text = normalize_text(textline['text'])
				key_type_corpus_map[fk_type][fk].update(list(text))
	return key_type_corpus_map


def get_corpus_stats(samples, required_keys):
	key_type_corpus_map = get_corpus_map(samples, required_keys)
	overall_corpus = get_common_corpus(key_type_corpus_map)
	print('Total: ', len(overall_corpus))
	for fk_type, key_corpus_map in sorted(key_type_corpus_map.items()):		
		fk_type_corpus = get_common_corpus(key_corpus_map)
		fk_type_corpus = fk_type_corpus.intersection(overall_corpus)
		print(' - ' + fk_type + ': ', len(fk_type_corpus))
		for fk, fk_corpus in sorted(key_corpus_map.items()):
			fk_corpus = fk_corpus.intersection(fk_type_corpus)
			print('   - ' + fk + ': ', len(fk_corpus), '(', ''.join(sorted(fk_corpus)) ,')')


def get_corpus_intersection_map(corpus_map):
	if type(corpus_map) is set:
		return {'#intersection' : corpus_map}
	
	res = {}
	intersection = None
	for key, sub_map in corpus_map.items():
		temp = get_corpus_intersection_map(sub_map)	
		if intersection is None:
			intersection = deepcopy(temp['#intersection'])
		else:
			intersection.intersection(temp['#intersection'])
		res[key] = temp

	if intersection is not None:
		for key in res:
			res[key]['#intersection'] = res[key]['#intersection'] - intersection 
	
	res['#intersection'] = intersection
	return res


def print_intersection_count_map(intersection_map, depth=0):
	current_intersection = intersection_map.pop('#intersection')
	cur_pad = ' ' * depth
	print(cur_pad + 'Total: ', len(current_intersection), '(', ''.join(sorted(current_intersection)) ,')')
	for key, sub_map in intersection_map.items():
		print(cur_pad + '- ' + key + ': ')
		print_intersection_count_map(sub_map, depth + 3)


def stats_corpus(samples, required_keys):	
	key_type_corpus_map = get_corpus_map(samples, required_keys)
	intersection_map = get_corpus_intersection_map(key_type_corpus_map)
	print_intersection_count_map(intersection_map)
	

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--path', help='Dataset directory path', required=True)	
	parser.add_argument('--classes', help='Classes file path', default=None)	
	parser.add_argument('--selected', nargs='+', help='Selected file names .lst paths', default=[])	
	
	args = parser.parse_args()
	data_dir_path = args.path
	required_keys = None
	class_path = args.classes
	if class_path:
		required_keys = load_json(class_path)
		logTracker.log('Loaded ' + str(len(required_keys)) + ' formal keys from ' + class_path)

	json_paths = loadValidFiles(data_dir_path, 'json', keepFilePath=True)
	json_paths = filter_file_paths_from_path_lists(json_paths, args.selected)
	stats_corpus(json_paths, required_keys=required_keys)
