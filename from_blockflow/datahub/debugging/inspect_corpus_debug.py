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

if __name__ == '__main__':
	project_dir = os.path.join('data', 'invoice')
	data_dir = os.path.join(project_dir, 'generated', '20190924-224204')
	img_dir = os.path.join(project_dir, 'stuff', 'imgs')
	
	qa_dir = os.path.join(data_dir, 'qa-labels-fk-mapped')
	label_json_paths = loadValidFiles(qa_dir, 'json', keepFilePath=True)

	#label_json_paths = sorted(label_json_paths)
	
	class_path = os.path.join(project_dir, 'classes.json')
	classes = load_json(class_path)

	corpus_path = os.path.join(data_dir, 'corpus-info-new', 'corpus.json')
	#corpus_path = os.path.join(data_dir, 'corpus-info', 'corpus.json')
	#corpus_path = os.path.join(project_dir, 'generated','20190822-150651', '20190823-023131-corpus', 'corpus.json')
	corpus = load_json(corpus_path)

	max_width = 1800
	content_type = 'text'
	class_color_map = get_class_color_map(classes)

	progress = ProgressBar('Inspecting', len(label_json_paths))
	all_coverage = defaultdict(lambda : [])
	for label_path in label_json_paths:
		sample = load_json(label_path)
		coverage = get_fk_corpus_coverage(sample, corpus)
		for key in coverage:
			all_coverage[key].append(coverage[key])
		progress.increase()
	progress.done()

	prefixs = set([key.split('#')[0] for key in all_coverage])
	prefixs = sorted(prefixs)

	for prefix in prefixs:
		print('Stats for "' + prefix + '"')
		table = BeautifulTable()
		table.column_headers = ['FK', 'count', 'min', 'average', '10pct', 'median', '90pct', 'max', 'std']
		for key in sorted(all_coverage):
			if not key.startswith(prefix + '#'):
				continue
			vals = all_coverage[key]
			stat_vals = [len(vals), np.min(vals), np.average(vals), np.percentile(vals, 10), np.median(vals), np.percentile(vals, 90), np.max(vals), np.std(vals)]
			stat_strs = [str(round(v, 5)) for v in stat_vals]
			table.append_row([key.replace(prefix + '#', '')] + stat_strs)
		print(table)
		print('')