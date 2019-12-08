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
from normalizing.normalize_text import normalize_text
from chi_lib.Multiprocessor import Multiprocessor
from chi_lib.ProgressBar import ProgressBar
from copy import deepcopy


def get_overlap_infos(target_ocr_textline, qa_textlines):
	if len(qa_textlines) == 0:
		return None, None
	cur_loc = get_topleft_bottomright(target_ocr_textline)
	locs = [get_topleft_bottomright(tl) for tl in qa_textlines]
	cur_area = get_rectangle_area(cur_loc)
	infos = []
	for i, loc in enumerate(locs):
		intersection_area = get_intersection_area(cur_loc, loc)
		overlapping = intersection_area / cur_area if cur_area > 1e-5 else 0
		temp_area = get_rectangle_area(loc)
		being_overlapped = intersection_area / temp_area if temp_area > 1e-5 else 0
		infos.append({
			'inner_overlap' : overlapping, 
			'outer_overlap' : being_overlapped, 
			'textline'		: qa_textlines[i]})
	return infos


def get_best_match_textline(target_ocr_textline, infos, corpus):
	for info in infos:
		info['common_bow'] = 0
	if not corpus is None:
		cur_bow = set(normalize_text(target_ocr_textline['text'])).intersection(corpus)
		for info in infos:
			qa_textline = info['textline']
			common_bow = set(normalize_text(qa_textline['text'])).intersection(cur_bow)
			info['common_bow'] = len(common_bow)

	best_match = max(infos, key=lambda x: (x['common_bow'], x['inner_overlap'], x['outer_overlap']))
	return best_match


def generate_labeled_ocr_sample(ocr_textlines, qa_textlines, required_classes=None, corpus=None):
	def is_match(info):
		label_info = info['textline']['label_info']
		if not label_info['key_type'] in ['value', 'key']:
			return False
		if required_classes != None:
			if not label_info['formal_key'] in required_classes:
				return False
		vals = [info['inner_overlap'], info['outer_overlap']]
		return min(vals) > 0.05 and max(vals) > 0.7

	if not corpus is None:
		corpus = set(corpus)

	qa_textlines = list(qa_textlines)
	ocr_textlines = deepcopy(ocr_textlines)
	for ocr_textline in ocr_textlines:
		infos = get_overlap_infos(ocr_textline, qa_textlines)
		infos = [info for info in infos if is_match(info)]
		if len(infos) > 0:
			best_match = get_best_match_textline(ocr_textline, infos, corpus)
			best_textline = best_match['textline']
			new_info = dict(best_textline['label_info'])
			new_info['text_gt'] = best_textline['text']
			new_info['inner_overlap'] = round(best_match['inner_overlap'], 4)
			new_info['outer_overlap'] = round(best_match['outer_overlap'], 4)
			ocr_textline['label_info'] = new_info
		else:
			ocr_textline['label_info'] = {'formal_key' : 'None', 'key_type' : 'other', 'inner_overlap' : 0, 'outer_overlap' : 0}
	return ocr_textlines


def label_ocr_samples(label_json_paths, ocr_json_paths, res_dir_path, required_classes, corpus):
	createDirectory(res_dir_path)
	qa_file_map = get_json_file_map_from_file_paths(label_json_paths)
	logTracker.log('Loaded ' + str(len(qa_file_map)) + ' QA samples')
	
	ocr_file_map = get_json_file_map_from_file_paths(ocr_json_paths)
	logTracker.log('Loaded ' + str(len(ocr_file_map)) + ' OCR samples')

	logTracker.log('Labeling for ' + str(len(ocr_file_map)) + ' OCR-Linecut samples')
	logTracker.log('Exporting to ' + res_dir_path)
	count = 0
	progress = ProgressBar(name='Labeling', maxValue=len(ocr_file_map))
	for f_name in ocr_file_map:
		progress.increase()
		if not f_name in qa_file_map:
			logTracker.log('Warning: Not found QA label file for "' + str(f_name) + '" with QA name "' + str(f_name) + '"')
			continue

		ocr_textlines = ocr_file_map[f_name]
		qa_textlines = qa_file_map[f_name]
		ocr_textlines = generate_labeled_ocr_sample(ocr_textlines, qa_textlines, required_classes, corpus)
		res_name = f_name.replace('.json', '_lc.json')
		with open(os.path.join(res_dir_path, res_name), 'w+', encoding='utf8') as f:
			f.write(json.dumps(ocr_textlines, ensure_ascii=False, indent=4))
			count += 1
	progress.done()
	logTracker.log('Total labeled files: ' + str(count) + ' / ' + str(len(ocr_file_map)))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--qa', help='Standard QA samples directory path', required=True)
	parser.add_argument('--ocr', help='Standard OCR-Linecut samples directory path', required=True)
	parser.add_argument('--classes', help='Classes file path', default=None)
	parser.add_argument('--corpus', help='Corpus file path', default=None)
	parser.add_argument('--selected', nargs='+', help='Selected file names .lst paths', default=[])
	args = parser.parse_args()

	label_dir_path = args.qa
	ocr_dir_path = args.ocr
	classes = None
	if args.classes:
		classes = load_json(args.classes)

	corpus = None
	if args.corpus:
		corpus = load_json(args.corpus)

	res_dir_path = os.path.join(getParentPath(ocr_dir_path), getBasename(ocr_dir_path) + '-labeled-' + getTodayDatetimeString())

	label_json_paths = loadValidFiles(label_dir_path, 'json', keepFilePath=True)
	ocr_json_paths = loadValidFiles(ocr_dir_path, 'json', keepFilePath=True)

	label_json_paths = filter_file_paths_from_path_lists(label_json_paths, args.selected)
	ocr_json_paths = filter_file_paths_from_path_lists(ocr_json_paths, args.selected)
	label_ocr_samples(label_json_paths, ocr_json_paths, res_dir_path, classes, corpus)
		



