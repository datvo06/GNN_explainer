# encoding: utf-8


import os, sys, json
sys.path.append('.') 

import numpy as np
import argparse
import shutil

from chi_lib.library import *


def standardize(json_paths, required_keys, key_map, mapping_func, res_dir):
	required_keys = set(required_keys)
	if key_map is None:
		logTracker.log('Not found mapping file\nUse one-to-one mapping by default')
		key_map = {fk : [fk] for fk in required_keys}
	mapping_func(json_paths, required_keys, key_map, res_dir)
	return required_keys, key_map


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--path', help='Dataset directory path', required=True)
	parser.add_argument('--res', help='Exporting directory path', required=True)
	parser.add_argument('--classes', help='Classes file path', required=True)
	parser.add_argument('--mapping', help='Classes mapping file path', required=True)
	parser.add_argument('--selected', nargs='+', help='Selected file names .lst paths', default=[])
	args = parser.parse_args()

	standard_raw_data_dir = args.path
	class_path = args.classes
	map_path = args.mapping

	cur_dir = getParentPath(standard_raw_data_dir)
	res_dir = os.path.join(cur_dir, getBasename(standard_raw_data_dir) + '-fk-standard-' + getTodayDatetimeString()) 

	required_keys = load_json(class_path)
	key_map = load_json(map_path, ignored_not_exists=True)	
	from processing.mapping_formal_key import mapping as mapping_func_default

	json_paths = loadValidFiles(standard_raw_data_dir, 'json', keepFilePath=True)
	json_paths = filter_file_paths_from_path_lists(json_paths, args.selected)
	standardize(json_paths, required_keys, key_map, mapping_func_default, res_dir)
	

