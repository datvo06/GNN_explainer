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


def get_standard_input_format(qaSample):	
	if not 'attributes' in qaSample:
		return None
	qaSample = qaSample['attributes']
	if '_via_img_metadata' in qaSample:
		qaSample = qaSample["_via_img_metadata"]

	res = []
	for region in qaSample['regions']:
		region_attr = region['region_attributes']
		shape_attr = region['shape_attributes']

		if shape_attr['name'] != 'rect':
			location = list(zip(shape_attr['all_points_x'], shape_attr['all_points_y']))
		else:
			x, y, w, h = shape_attr['x'], shape_attr['y'], shape_attr['width'], shape_attr['height']
			location = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]	

		if 'type' in region_attr: 
			cur_type = region_attr['type']
		else:
			cur_type = region_attr['key_type']

		label_info = {
			'formal_key'	: region_attr['formal_key'].strip('\n\t\r '),
			'key_type'		: str(cur_type)
		}
		ignored = ['label', 'formal_key', 'type', 'key_type']
		# Retain the other information
		for attr in region_attr:
			if not attr in label_info and not attr in ignored:
				try:
					label_info[attr] = region_attr[attr].strip('\n\t\r ')
				except AttributeError:
					label_info[attr] = region_attr[attr]

		temp = {
			'location'		: location,
			'type' 			: 'textline',
			'text'			: str(region_attr['label']),
			'label_info'	: label_info
		}
		
		res.append(temp)
	return res


def standardize(json_paths, resDirPath):
	createDirectory(resDirPath)
	logTracker.log('Standardizing ' + str(len(json_paths)) + ' QA samples to ' + resDirPath)
	progress = ProgressBar(name='Converting', maxValue=len(json_paths))
	for fPath in json_paths:
		try:
			# with open(fPath, 'r', encoding='utf8') as f:
			with open(fPath, 'r', encoding='utf-8-sig') as f:
				newJson = json.loads(f.read())
			newJson = get_standard_input_format(newJson)
			if newJson is None:
				logTracker.log('Cannot convert ' + fPath)
				continue

			resPath = os.path.join(resDirPath, getBasename(fPath))
			with open(resPath, 'w+', encoding='utf8') as f:
				f.write(json.dumps(newJson, ensure_ascii=False, indent=4))
		except Exception as e:
			traceback.print_exc()
			logTracker.logException(str(e) + '\n\n' + str('File path: ' + fPath))
		progress.increase()
	progress.done()
	return resDirPath


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--path', help='Raw labels directory path', required=True)
	parser.add_argument('--selected', nargs='+', help='Selected file names .lst paths', default=[])
	parser.add_argument('--res', help='Exporting directory path', default=None)
	args = parser.parse_args()

	raw_qa_dir_path = args.path
	resDirPath = args.res
	if resDirPath is None:
		curDirPath = getParentPath(raw_qa_dir_path)
		resDirPath = os.path.join(curDirPath, QA_STANDARD_LABELS_DIR)

	json_paths = loadValidFiles(raw_qa_dir_path, 'json', keepFilePath=True)
	json_paths = filter_file_paths_from_path_lists(json_paths, args.selected)
	standardize(json_paths, resDirPath)

