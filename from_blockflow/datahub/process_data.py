# encoding: utf-8

import os, sys, json
import glob                                                           
import cv2 
import numpy as np
import argparse
import shutil
import re

from collections import defaultdict
from frozendict import frozendict

from chi_lib.Multiprocessor import Multiprocessor
from chi_lib.ProgressBar import ProgressBar
from chi_lib.library import *
from directory_name import *

from normalizing.rename_standdard_format import rename_standard_from_dir
from normalizing.standardize_ocr_output import standardize as ocr_standardize
from normalizing.standardize_qa_labels import standardize as qa_standardize
from normalizing.standardize_formal_keys import standardize as fk_standardize
from processing.label_ocr_samples import label_ocr_samples
from processing.generate_corpus import generate_corpus, export_corpus_coverage_stats



def check_conflict(ocr_rename_map, qa_rename_map):
	logTracker.log('Warning: Did not check conflict for file name mapping')
	pass


def mix_ocr_samples(generated_dir_path, ocr_raw_json_paths, qa_fk_res_json_paths, required_keys, corpus, mix_dir_path):
	logTracker.log('Mixing OCR/Line labeled samples with QA samples')
	ocr_res_path = os.path.join(generated_dir_path, OCR_STANDARD_OUTPUT_DIR + '-standard')
	ocr_standardize(ocr_raw_json_paths, ocr_res_path)

	ocr_labels_res_dir_path = os.path.join(generated_dir_path, OCR_LABELS_DIR)
	ocr_json_paths = loadValidFiles(ocr_res_path, 'json', keepFilePath=True)
	label_ocr_samples(qa_fk_res_json_paths, ocr_json_paths, ocr_labels_res_dir_path, required_classes=required_keys, corpus=corpus)

	ocr_labels_res_json_paths = loadValidFiles(ocr_labels_res_dir_path, 'json', keepFilePath=True)
	json_paths = ocr_labels_res_json_paths + qa_fk_res_json_paths
	copy_files(json_paths, mix_dir_path)


def generated_standard_samples(project_data_dir, generated_dir_path, is_normalize, max_corpus_size, mapping_func=None, mix_ocr=True, lst_selected_paths=[], corpus=None):
	createDirectory(generated_dir_path)

	# Standardizing file names
	qa_raw_path = os.path.join(project_data_dir, RAW_DATA_DIR, QA_STANDARD_LABELS_DIR)
	qa_raw_json_paths = loadValidFiles(qa_raw_path, 'json', keepFilePath=True)
	qa_raw_json_paths = filter_file_paths_from_path_lists(qa_raw_json_paths, lst_selected_paths)
	if is_normalize:
		qa_rename_map = rename_standard_from_dir(qa_raw_json_paths)
		qa_raw_json_paths = list(qa_rename_map.values())

	# Standarizing QA samples
	qa_res_path = os.path.join(generated_dir_path, QA_STANDARD_LABELS_DIR + '-standard')
	qa_standardize(qa_raw_json_paths, qa_res_path)

	# Getting required information for formal key normalization
	qa_fk_res_dir = os.path.join(generated_dir_path, QA_STANDARD_LABELS_DIR + '-fk-mapped') 
	class_path = os.path.join(project_data_dir, CLASSES_FILE)
	map_path = os.path.join(project_data_dir, LABELS_MAPPING_FILE)
	# 'mapping_func' must be a callable module
	if mapping_func is None:
		from processing.mapping_formal_key import mapping as mapping_func_default
		mapping_func = mapping_func_default

	required_keys = load_json(class_path)
	key_map = load_json(map_path, ignored_not_exists=True)

	# Normalizing formal keys
	qa_res_json_paths = loadValidFiles(qa_res_path, 'json', keepFilePath=True)
	required_keys, key_map = fk_standardize(qa_res_json_paths, required_keys, key_map, mapping_func, qa_fk_res_dir)

	qa_fk_res_json_paths = loadValidFiles(qa_fk_res_dir, 'json', keepFilePath=True)

	corpus_res_path = os.path.join(generated_dir_path, CORPUS_INFO_DIR + '-qa')
	createDirectory(corpus_res_path)
	qa_samples = load_jsons(qa_fk_res_json_paths)
	if corpus is None:
		corpus = generate_corpus(qa_samples, max_corpus_size, required_keys=required_keys)
	save_json(corpus, os.path.join(corpus_res_path, 'corpus.json'))	
	export_corpus_coverage_stats(qa_samples, corpus, corpus_res_path)

	cur_samples_dir = qa_fk_res_dir
	if mix_ocr is True:
		# Standardizing file names and check file name conflict
		ocr_raw_path = os.path.join(project_data_dir, RAW_DATA_DIR, OCR_STANDARD_OUTPUT_DIR)
		ocr_raw_json_paths = loadValidFiles(ocr_raw_path, 'json', keepFilePath=True)
		ocr_raw_json_paths = filter_file_paths_from_path_lists(ocr_raw_json_paths, lst_selected_paths)
	
		if is_normalize:
			ocr_rename_map = rename_standard_from_dir(ocr_raw_json_paths)
			check_conflict(ocr_rename_map, qa_rename_map)
			ocr_raw_json_paths = list(ocr_rename_map.values())

		# Standardizing and mixing the OCR/Linecut samples with QA samples
		mix_dir_path = os.path.join(generated_dir_path, MIXED_LABELS_DIR)
		corpus = load_json(os.path.join(corpus_res_path, 'corpus.json'))
		mix_ocr_samples(generated_dir_path, ocr_raw_json_paths, qa_fk_res_json_paths, required_keys, corpus, mix_dir_path)
		cur_samples_dir = mix_dir_path

	return cur_samples_dir


def process_data(project_data_dir, generated_dir_path, samples_dir, corpus, max_corpus_size):
	json_paths = loadValidFiles(samples_dir, 'json', keepFilePath=True)
	res_path = os.path.join(generated_dir_path, CORPUS_INFO_DIR)

	class_path = os.path.join(project_data_dir, CLASSES_FILE)
	required_keys = load_json(class_path)

	samples = load_jsons(json_paths)
	if corpus is None:
		corpus = generate_corpus(samples, max_corpus_size, required_keys=required_keys)
	
	corpus_res_path = os.path.join(generated_dir_path, CORPUS_INFO_DIR + '-all')
	createDirectory(corpus_res_path)
	save_json(corpus, os.path.join(corpus_res_path, 'corpus.json'))	
	export_corpus_coverage_stats(samples, corpus, corpus_res_path)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--path', help='Project data directory path', required=True)
	parser.add_argument('--name_norm', help='Normalize file names to match between OCR/Linecut samples and QA samples', action='store_true')
	parser.add_argument('--corpus', help='Predefined corpus file path', default=None)
	parser.add_argument('--max_corpus', help='Maximum length of the corpus', default=500)
	parser.add_argument('--not_mix_ocr', help='Do not mix OCR samples', action='store_false')
	parser.add_argument('--selected', nargs='+', help='Selected file names .lst paths', default=[])	
	args = parser.parse_args()

	project_data_dir = args.path
	generated_dir_path = os.path.join(project_data_dir, GENERATED_DIR, getTodayDatetimeString())
	is_normalize = args.name_norm
	is_mix_ocr = args.not_mix_ocr
	max_corpus_size = int(args.max_corpus)
	
	corpus = None
	if args.corpus:
		corpus = load_json(args.corpus)

	mapping_func = None
	if 'invoice' in getBasename(project_data_dir):
		from processing.mapping_formal_key_invoice import mapping as mapping_func_invoice
		mapping_func = mapping_func_invoice

	standard_samples_dir = generated_standard_samples(project_data_dir, generated_dir_path, is_normalize, max_corpus_size=max_corpus_size, mapping_func=mapping_func, mix_ocr=is_mix_ocr, lst_selected_paths=args.selected, corpus=corpus)

	process_data(project_data_dir, generated_dir_path, standard_samples_dir, corpus=corpus, max_corpus_size=max_corpus_size)
