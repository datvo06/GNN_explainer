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
from visualizing.stats_corpus_coverage import generate_coverage_stats


def get_corpus_map(samples):
	logTracker.log('Getting corpus map for ' + str(len(samples)) + ' samples')

	fk_count_map = defaultdict(lambda : defaultdict(lambda : 0))
	progress = ProgressBar('Counting', len(samples))
	for sample in samples:
		for textline in sample:
			label_info = textline['label_info']
			fk = label_info['formal_key']
			fk_type = label_info['key_type']
			fk_count_map[fk_type][fk] += 1
		progress.increase()
	progress.done()
	
	corpus_map = defaultdict(lambda : defaultdict(lambda : defaultdict(lambda : 0)))	
	progress = ProgressBar('Getting', len(samples))
	for sample in samples:
		for textline in sample:
			label_info = textline['label_info']
			fk = label_info['formal_key']
			fk_type = label_info['key_type']
			text = textline['text']
			text = set(text)
			for c in text:
				corpus_map[fk_type][fk][c] += (1 / len(text)) * (1 / fk_count_map[fk_type][fk])
		progress.increase()
	progress.done()
	return corpus_map


def get_fk_corpus(fk_corpus_map, required_keys=None):
	corpus = set()
	for fk, fk_data in fk_corpus_map.items():
		if required_keys is None or fk in required_keys:
			corpus.update(list(fk_data.keys()))
	return corpus


def add_obligated_corpus(corpus):
	#temp = [chr(x) for x in range(ord('a'), ord('z') + 1)]
	temp = []
	temp = temp + ['0', ',', '.'] 

	# This step is to remove all the character, which we want to put it on the starting of the corpus list
	temp_corpus = []
	for c in corpus:
		if not c in temp:
			temp_corpus.append(c)

	# Now, we add each obligated character to the beginning of the list
	corpus = temp_corpus
	for c in temp:
		corpus = [c] + corpus
	return corpus


def generate_corpus(samples, max_size, required_keys=None):
	logTracker.log('Generating corpus with size ' + str(max_size) + ' from ' + str(len(samples)) + ' samples')

	progress = ProgressBar(name='Normalizing', maxValue=len(samples))
	for sample in samples:
		for textline in sample:
			textline['text'] = normalize_text(textline['text'])
		progress.increase()
	progress.done()

	corpus_map = get_corpus_map(samples)
	key_corpus = get_fk_corpus(corpus_map['key'], required_keys)
	value_corpus = get_fk_corpus(corpus_map['value'], required_keys)

	corpus = add_obligated_corpus(key_corpus)
	value_corpus_score = defaultdict(lambda : 0)
	for fk, fk_data in corpus_map['value'].items():
		for c in fk_data:
			value_corpus_score[c] += fk_data[c]

	value_corpus = sorted(value_corpus_score.keys(), key=lambda x: value_corpus_score[x], reverse=True)
	for c in value_corpus:
		if not c in corpus:
			corpus += c
	corpus = corpus[:max_size]
	corpus = ''.join(sorted(set(corpus)))
	return corpus


def export_corpus_coverage_stats(samples, corpus, res_dir):
	stats = generate_coverage_stats(samples, corpus)
	stats_dir = os.path.join(res_dir, 'debug')
	createDirectory(stats_dir)
	logTracker.log('Saving stats to ' + stats_dir)
	for prefix in stats:
		with open(os.path.join(stats_dir, prefix + '.txt'), 'w+', encoding='utf8') as f:
			f.write(stats[prefix])

	
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--path', help='Dataset directory path', required=True)	
	parser.add_argument('--classes', help='Classes file path', default=None)	
	parser.add_argument('--res', help='Exporting file path', default=None)	
	parser.add_argument('--selected', nargs='+', help='Selected file names .lst paths', default=[])	
	
	args = parser.parse_args()
	data_dir_path = args.path
	res_path = args.res
	if res_path is None:
		res_path = os.path.join(getParentPath(data_dir_path), getTodayDatetimeString() + '-corpus')

	required_keys = None
	class_path = args.classes
	if class_path:
		required_keys = load_json(class_path)
		logTracker.log('Loaded ' + str(len(required_keys)) + ' formal keys from ' + class_path)

	json_paths = loadValidFiles(data_dir_path, 'json', keepFilePath=True)
	json_paths = filter_file_paths_from_path_lists(json_paths, args.selected)
	samples = load_jsons(json_paths)
	corpus = generate_corpus(samples, max_size=600, required_keys=required_keys)
	save_json(corpus, os.path.join(res_path, 'corpus.json'))

	stats = generate_coverage_stats(samples, corpus)
	stats_dir = os.path.join(res_path, 'debug')
	createDirectory(stats_dir)
	logTracker.log('Saving stats to ' + stats_dir)
	for prefix in stats:
		with open(os.path.join(stats_dir, prefix + '.txt'), 'w+', encoding='utf8') as f:
			f.write(stats[prefix])
