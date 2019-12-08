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
from directory_name import OCR_STANDARD_OUTPUT_DIR


def convert_to_new_format(ocr_json):
	def get_new_location_format(old_format):
		if not type(old_format[0]) is list:
			x1 = int(old_format[0])
			y1 = int(old_format[1])
			x2 = int(old_format[2])
			y2 = int(old_format[3])
			return ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
		return old_format

	def get_text(old_format):
		if 'text' in old_format:
			text = old_format['text']
		else:
			text = old_format['value']
		return str(text)

	if type(ocr_json) is dict:
		ocr_json = list(ocr_json.values())

	res = []
	for tl in ocr_json:
		res.append({
			'location' 	: get_new_location_format(tl['location']),
			'type'		: 'textline',
			'text'		: get_text(tl)
		})
	return res


def standardize(json_paths, resDirPath):
	createDirectory(resDirPath)
	logTracker.log('Standardizing ' + str(len(json_paths)) + ' OCR-Linecut samples to ' + resDirPath)
	progress = ProgressBar(name='Converting', maxValue=len(json_paths))
	for fPath in json_paths:
		ocr_json = load_json(fPath)
		new_json = convert_to_new_format(ocr_json)
		new_name = getBasename(fPath)
		with open(os.path.join(resDirPath, new_name), 'w+', encoding='utf8') as f:
			f.write(json.dumps(new_json, ensure_ascii=False, indent=4))
		progress.increase()
	progress.done()	

	return resDirPath


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--path', help='Raw OCR-Linecut dataset directory path', required=True)
	parser.add_argument('--selected', nargs='+', help='Selected file names .lst paths', default=[])	
	parser.add_argument('--res', help='Exporting directory path', default=None)
	#parser.add_argument('--not_normalize', help='Do not normalize file name', action='store_false')
	args = parser.parse_args()

	raw_ocr_dir_path = args.path
	resDirPath = args.res
	if resDirPath is None:
		curDirPath = getParentPath(raw_ocr_dir_path)
		resDirPath = os.path.join(curDirPath, OCR_STANDARD_OUTPUT_DIR)

	json_paths = loadValidFiles(raw_ocr_dir_path, 'json', keepFilePath=True)	
	json_paths = filter_file_paths_from_path_lists(json_paths, args.selected)
	standardize(json_paths, resDirPath)