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
from copy import deepcopy

from normalizing.normalize_text import normalize_text
from processing.label_ocr_samples import generate_labeled_ocr_sample
from visualizing.visualize import visualize_samples, get_class_color_map
from beautifultable import BeautifulTable


def get_coverage(cur_text, corpus):
	count = 0
	if len(cur_text) == 0:
		return 0
	cur_text = normalize_text(cur_text)
	cur_text = set(cur_text)
	for c in cur_text:
		if c in corpus:
			count += 1
	return float(count) / len(cur_text)


def get_fk_corpus_coverage(sample, corpus):
	res = defaultdict(lambda : [])
	for textline in sample:
		label_info = textline['label_info']
		fk = label_info['formal_key']
		fk_type = label_info['key_type']
		res[fk_type + '#' + fk].append(get_coverage(textline['text'], corpus))
	for key in res:
		res[key] = np.mean(res[key])
	return dict(res)


def generate_coverage_stats(samples, corpus):
	progress = ProgressBar('Inspecting', len(samples))
	all_coverage = defaultdict(lambda : [])
	for sample in samples:
		coverage = get_fk_corpus_coverage(sample, corpus)
		for key in coverage:
			all_coverage[key].append(coverage[key])
		progress.increase()
	progress.done()

	prefixs = set([key.split('#')[0] for key in all_coverage])
	prefixs = sorted(prefixs)

	res = {}
	for prefix in prefixs:
		message = 'Stats for "' + prefix + '"'
		table = BeautifulTable()
		table.column_headers = ['FK', 'count', 'min', 'average', '10pct', 'median', '90pct', 'max', 'std']
		for key in sorted(all_coverage):
			if not key.startswith(prefix + '#'):
				continue
			vals = all_coverage[key]
			stat_vals = [len(vals), np.min(vals), np.average(vals), np.percentile(vals, 10), np.median(vals), np.percentile(vals, 90), np.max(vals), np.std(vals)]
			stat_strs = [str(round(v, 5)) for v in stat_vals]
			table.append_row([key.replace(prefix + '#', '')] + stat_strs)
		res[prefix] = str(table)
	return res


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--path', help='Dataset directory path', required=True)	
	parser.add_argument('--corpus', help='Corpus file path', required=True)
	parser.add_argument('--classes', help='Classes file path', default=None)
	parser.add_argument('--selected', nargs='+', help='Selected file names .lst paths', default=[])	
	
	args = parser.parse_args()
	data_dir_path = args.path

	required_keys = None
	class_path = args.classes
	if class_path:
		required_keys = load_json(class_path)
		logTracker.log('Loaded ' + str(len(required_keys)) + ' formal keys from ' + class_path)

	corpus = load_json(args.corpus)
	logTracker.log('Loaded corpus with size ' + str(len(corpus)))

	json_paths = loadValidFiles(data_dir_path, 'json', keepFilePath=True)
	json_paths = filter_file_paths_from_path_lists(json_paths, args.selected)

	samples = load_jsons(json_paths)
	res = generate_coverage_stats(samples, corpus)
	for fk_type in res:
		print('Stats for ' + fk_type)
		print(res[fk_type])
		print('')
