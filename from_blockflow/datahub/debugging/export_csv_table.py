# encoding: utf-8


import os, sys, json
sys.path.append('.') 

import glob                                                           
import cv2 
import numpy as np
import argparse
import shutil
import traceback

from chi_lib.library import *
from collections import defaultdict
from frozendict import frozendict
from chi_lib.Multiprocessor import Multiprocessor
from chi_lib.ProgressBar import ProgressBar
from directory_name import QA_STANDARD_LABELS_DIR


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--path', help='Raw labels directory path', required=True)
	parser.add_argument('--selected', nargs='+', help='Selected file names .lst paths', default=[])
	args = parser.parse_args()

	raw_qa_dir_path = args.path
	curDirPath = getParentPath(raw_qa_dir_path)
	resDirPath = os.path.join(curDirPath, QA_STANDARD_LABELS_DIR)

	json_paths = loadValidFiles(raw_qa_dir_path, 'json', keepFilePath=True)
	json_paths = filter_file_paths_from_path_lists(json_paths, args.selected)
	
	res = []
	common_keys = set()
	for json_path in json_paths:
		temp = load_json(json_path)
		sample_dict = defaultdict(lambda : [])
		sample_dict['_file_name'].append(getBasename(json_path))
		for textline in temp:
			label_info = textline['label_info']
			fk = label_info['formal_key']
			fk_type = label_info['key_type']
			sample_dict[fk_type + '_' + fk].append(textline['text'])
		res.append(sample_dict)
		common_keys.update(list(sample_dict.keys()))

	delimiter = '\t'
	common_keys = sorted(common_keys)
	with open(os.path.join(curDirPath, getBasename(raw_qa_dir_path) + '.csv'), 'w+', encoding='utf8') as f:
		f.write(mergeList(common_keys, delimiter) + '\n')
		for sample_dict in res:
			temp = []
			for key in common_keys:
				temp.append(mergeList(sorted(sample_dict[key]), ' | '))
			f.write(mergeList(temp, delimiter) + '\n')