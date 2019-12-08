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

from processing.mapping_formal_key_invoice import mapping


if __name__ == '__main__':
	project_dir = os.path.join('data', 'invoice')
	data_dir = os.path.join(project_dir, 'generated', '20190829-120006')

	qa_dir = os.path.join(data_dir, 'qa-labels-standard')
	file_name = '0785_039_27.json'
	label_json_paths = [os.path.join(qa_dir, file_name)]
	for f_path in label_json_paths:
		textlines = load_json(f_path)
		for textline in textlines:
			if '045-641' in textline['text']:
				print(textline)
	
	classes = load_json(os.path.join(project_dir, 'classes.json'))
	key_map = load_json(os.path.join(project_dir, 'labels-mapping.json'))

	res_dir_path = os.path.join('debugging', 'fk_mapping_debug')
	mapping(label_json_paths, classes, key_map, res_dir_path)
		
	label_json_paths = [os.path.join(res_dir_path, file_name)]
	for f_path in label_json_paths:
		textlines = load_json(f_path)
		for textline in textlines:
			if '045-641' in textline['text']:
				print(textline)


