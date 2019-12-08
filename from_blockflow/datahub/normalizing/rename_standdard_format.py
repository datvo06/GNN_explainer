# encoding: utf-8


import os, sys, json
sys.path.append('.') 

import glob                                                           
import cv2 
import numpy as np
import argparse
import shutil
import re

from chi_lib.library import *
from collections import defaultdict
from frozendict import frozendict
from chi_lib.Multiprocessor import Multiprocessor
from chi_lib.ProgressBar import ProgressBar
from directory_name import *


replace_pattern = re.compile(r'\.png\.|\.jpg\.|\.tif\.', re.IGNORECASE)
def rename_standard(f_path):
	cur_dir = getParentPath(f_path)
	new_name = getBasename(f_path)
	new_name = replace_pattern.sub('.', new_name)	
	new_name = new_name.replace('_flax_lc.json', '.json')
	new_name = new_name.replace('_flax.json', '.json')
	new_name = new_name.replace('_lc.json', '.json')
	res_path = os.path.join(cur_dir, new_name)
	os.rename(f_path, res_path)
	return res_path


def rename_standard_from_dir(json_paths):
	rename_map = {}
	logTracker.log('Renaming ' + str(len(json_paths)) + ' files')
	progress = ProgressBar('Renaming', len(json_paths))
	for f_path in json_paths:
		new_path = rename_standard(f_path)
		rename_map[f_path] = new_path
		progress.increase()
	progress.done()
	existing_paths = set()
	for f_path in rename_map:
		new_path = rename_map[f_path]
		if new_path in existing_paths:
			logTracker.log('Warning, duplicated json path: ' + str(new_path))
		existing_paths.add(new_path)
	logTracker.log('Renamed ' + str(len(existing_paths)) + ' json files')
	return rename_map


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--path', help='QA or OCR-Linecut dataset directory path', required=True)
	parser.add_argument('--selected', nargs='+', help='Selected file names .lst paths', default=[])
	args = parser.parse_args()

	data_dir_path = args.path
	json_paths = loadValidFiles(data_dir_path, 'json', keepFilePath=True)
	json_paths = filter_file_paths_from_path_lists(json_paths, args.selected)
	rename_standard_from_dir(json_paths)